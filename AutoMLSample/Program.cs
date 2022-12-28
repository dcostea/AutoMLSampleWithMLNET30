using AutoMLSample.Services;

Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .MinimumLevel.Override("Microsoft", Serilog.Events.LogEventLevel.Warning)
            .MinimumLevel.Override("System", Serilog.Events.LogEventLevel.Warning)
            .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}")
            .WriteTo.File("logs/check.log", Serilog.Events.LogEventLevel.Debug, "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}{Exception}")
            .CreateBootstrapLogger();


////string DatasetPath = "Data/sensors_data.csv";
////const string Label = "Source";

string DatasetPath = "Data/titanic.csv";
const string Label = "Survived";

(var trainTestData, var columnInference) = MachineLearningServices.LoadDataAndColumns(DatasetPath, Label);

var experimentResult = await MachineLearningServices.AutoTrainAsync(60, trainTestData.TrainSet, columnInference);
var model = experimentResult.Model as TransformerChain<ITransformer>;

var transformedTestingData = model.Transform(trainTestData.TestSet);
var transformedData = model.Transform(trainTestData.TestSet);
MachineLearningServices.Evaluate(transformedTestingData, columnInference.ColumnInformation.LabelColumnName, showsConfusionMatrix: false);
MachineLearningServices.PFI(0.01F, model.LastTransformer, transformedData, columnInference.ColumnInformation.LabelColumnName);
MachineLearningServices.CorrelationMatrix(0.9F, trainTestData.TrainSet);
