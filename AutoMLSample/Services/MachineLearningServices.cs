using System.Data;
using MathNet.Numerics.Statistics;
using AutoMLSample.Models;
using Microsoft.ML.Trainers.FastTree;
using static AutoMLSample.Helpers.ConsoleHelpers;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Runtime;

namespace AutoMLSample.Services;

internal static class MachineLearningServices
{
    private const string Score = nameof(ModelOutput.Score);
    private const string PredictedLabel = nameof(ModelOutput.PredictedLabel);

    private static MLContext Context { get; set; } = new MLContext(seed: 1);

    internal static (TrainTestData, ColumnInferenceResults) LoadDataAndColumns(string datasetPath, string label)
    {
        var columnInference = Context.Auto().InferColumns(datasetPath, labelColumnName: label, groupColumns: false);

        // manually modify type of the columns below

        ////columnInference.ColumnInformation.TextColumnNames.Remove("CreatedAt");
        ////columnInference.ColumnInformation.IgnoredColumnNames.Add("CreatedAt");
        ////columnInference.ColumnInformation.NumericColumnNames.Remove("Temperature2");
        ////columnInference.ColumnInformation.IgnoredColumnNames.Add("Temperature2");

        columnInference.ColumnInformation.NumericColumnNames.Remove("Pclass");
        columnInference.ColumnInformation.NumericColumnNames.Remove("SibSp");
        columnInference.ColumnInformation.NumericColumnNames.Remove("Parch");
        columnInference.ColumnInformation.CategoricalColumnNames.Add("Pclass");
        columnInference.ColumnInformation.CategoricalColumnNames.Add("SibSp");
        columnInference.ColumnInformation.CategoricalColumnNames.Add("Parch");

        var loader = Context.Data.CreateTextLoader(columnInference.TextLoaderOptions);
        var data = loader.Load(datasetPath);
        
        TrainTestData trainValidationData = Context.Data.TrainTestSplit(data, testFraction: 0.3); // split into train (70%), validation (30%) sets

        return (trainValidationData, columnInference);
    }

    internal static async Task<TrialResult> AutoTrainAsync(uint time, IDataView data, ColumnInferenceResults columnInference)
    {
        Context.Log += (_, e) =>
        {
            if (e.Source.Equals("AutoMLExperiment") && e.Kind.Equals(ChannelMessageKind.Info))
            {
                WriteLineColor(e.Message, ConsoleColor.White);
            }
        };

        WriteLineColor($" INCREASE ML MODEL ACCURACY IN THREE STEPS");
        WriteLineColor($" Learning type: multi-class classification");
        WriteLineColor($" Training time: {time} seconds");
        WriteLineColor("----------------------------------------------------------------------------------");

        Context = new MLContext(seed: 1);
        
        SweepablePipeline preprocessingPipeline = Context.Transforms
            .Conversion.MapValueToKey(columnInference.ColumnInformation.LabelColumnName, columnInference.ColumnInformation.LabelColumnName)
            .Append(Context.Auto().Featurizer(data, columnInformation: columnInference.ColumnInformation));

        var pipeline = preprocessingPipeline
            .Append(Context.Auto().MultiClassification(labelColumnName: columnInference.ColumnInformation.LabelColumnName));

        AutoMLExperiment experiment = Context.Auto()
            .CreateExperiment()
            .SetPipeline(pipeline)
            .SetMulticlassClassificationMetric(MulticlassClassificationMetric.MicroAccuracy, labelColumn: columnInference.ColumnInformation.LabelColumnName)
            .SetTrainingTimeInSeconds(time)
            .SetDataset(data);

        // Log experiment trials
        var monitor = new AutoMLMonitor(pipeline);
        experiment.SetMonitor(monitor);

        ////var cts = new CancellationTokenSource();
        ////var experimentResult = await experiment.RunAsync(cts.Token);
        var experimentResult = await experiment.RunAsync();
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" STEP 1: AutoML experiment result");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" Best trainer: {monitor.GetBestTrial(experimentResult)}");
        WriteLineColor($" Accuracy: {experimentResult.Metric,-6:F3}    Training time: {experimentResult.DurationInMilliseconds,5}ms    CPU: {monitor.PeakCpu,5:P2}    Memory: {monitor.PeakMemoryInMegaByte,5:F2}MB");
        ////var completedTrials = monitor.GetCompletedTrials();

        return experimentResult;
    }

    internal static void Evaluate(IDataView transformedData, string label, bool showsConfusionMatrix)
    {
        var metrics = Context.MulticlassClassification.Evaluate(transformedData, label, Score, PredictedLabel);
        PrintMultiClassClassificationMetrics(metrics, showsConfusionMatrix);
    }

    internal static void PFI(float threshold, ITransformer transformer, IDataView transformedData, string label)
    {
        WriteLineColor(" STEP 2: PFI (permutation feature importance)");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($" PFI (by MicroAccuracy), threshold: {threshold}");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($"  {"No",4} {"Feature",-15} {"MicroAccuracy",15} {"95% Mean",15}");

        Context.Log += (_, e) =>
        {
            if (e.Source.StartsWith("Permutation") && e.Kind.Equals(ChannelMessageKind.Info))
            {
                WriteLineColor(e.Message, ConsoleColor.White);
            }
        };

        var pfi = Context.MulticlassClassification.PermutationFeatureImportance(transformer, transformedData, label, permutationCount: 5);
        ////var metrics = pfi.Select(p => (p.Key, p.Value.MicroAccuracy)).OrderBy(m => m.MicroAccuracy.Mean);

        // patching dot issue
        var patchedPfi = pfi.Select(p => new KeyValuePair<string, MulticlassClassificationMetricsStatistics>(p.Key.Split(".").First(), p.Value));
        var groupedPfi = patchedPfi.GroupBy(p => p.Key).ToDictionary(g => g.Key, g => g.Select(x => x.Value));
        var metrics = groupedPfi.Select(p => new MicroAccuracyModel
        {
            Key = p.Key,
            MicroAccuracy = new MetricStatisticsModel
            {
                Mean = p.Value.Sum(m => m.MicroAccuracy.Mean),
                StandardError = p.Value.Sum(m => m.MicroAccuracy.StandardError)
            }
        })
        .OrderBy(m => m.MicroAccuracy.Mean);

        uint noCnt = 1;
        foreach (var metric in metrics)
        {
            if (Math.Abs(metric.MicroAccuracy.Mean) < threshold)
            {
                WriteLineColor($"  {noCnt++,3}. {metric.Key,-15} {metric.MicroAccuracy.Mean,15:F5} {1.95 * metric.MicroAccuracy.StandardError,15:F5} (candidate for deletion!)", ConsoleColor.Red);
            }
            else
            {
                WriteLineColor($"  {noCnt++,3}. {metric.Key,-15} {metric.MicroAccuracy.Mean,15:F4} {1.95 * metric.MicroAccuracy.StandardError,15:F4}");
            }
        }
        WriteLineColor("----------------------------------------------------------------------------------");
    }

    //TODO use infered columns
    internal static void CorrelationMatrix(float threshold, IDataView trainingDataView)
    {
        var trainingDataCollection = Context.Data.CreateEnumerable<ModelInput>(trainingDataView, reuseRowObject: true);
        (var header, var dataArray) = trainingDataCollection.ExtractDataAndHeader();
        var matrix = Correlation.PearsonMatrix(dataArray.ToArray());
        ////var header = columnInference.ColumnInformation.NumericColumnNames
        ////    .Union(columnInference.ColumnInformation.CategoricalColumnNames)
        ////    .ToArray();
        matrix.ToConsole(header, threshold);

        WriteLineColor("  We can remove one of the next high correlated features!");
        WriteLineColor("    - closer to  0 => low correlated features");
        WriteLineColor("    - closer to  1 => direct high correlated features");
        WriteLineColor("    - closer to -1 => inverted high correlated features");
        WriteLineColor("----------------------------------------------------------------------------------");
        WriteLineColor($"  {"No",4} {"Feature",-15} vs. {"Feature",-15} {"Rate",15}");
        uint noCnt = 1;
        for (int i = 0; i < matrix.ColumnCount; i++)
        {
            for (int j = i; j < matrix.ColumnCount; j++)
            {
                if (i != j && Math.Abs(matrix[i, j]) > threshold)
                {
                    WriteLineColor($"  {noCnt++,3}. {header[i],-15} vs. {header[j],-15} {matrix[i, j],15:F4}", ConsoleColor.Red);
                }
            }
        }
        WriteLineColor("----------------------------------------------------------------------------------");
    }

    private static (string[] header, double[][] dataArray) ExtractDataAndHeader(this IEnumerable<ModelInput> trainingDataCollection)
    {
        var record = new ModelInput();
        var props = record.GetType().GetProperties();

        var data = new List<List<double>>();
        uint k = 0;
        foreach (var prop in props)
        {
            if (props[k].PropertyType.Name.Equals(nameof(Single)))
            {
                var arr = trainingDataCollection.Select(r => (double)(props[k].GetValue(r) as float?).Value).ToList();
                data.Add(arr);
            }
            k++;
        }
        var header = props.Where(s => s.PropertyType.Name.Equals(nameof(Single))).Select(p => p.Name).ToArray();
        var dataArray = new double[data.Count][];
        for (int i = 0; i < data.Count; i++)
        {
            dataArray[i] = data[i].ToArray();
        }

        return (header, dataArray);
    }

    internal class MicroAccuracyModel
    {
        public string Key { get; set; }
        public MetricStatisticsModel MicroAccuracy { get; set; }
    }

    internal class MetricStatisticsModel
    {
        public double Mean { get; set; }
        public double StandardError { get; set; }
        //public double StandardDeviation { get; set; }
    }
}
