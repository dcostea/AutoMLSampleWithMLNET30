﻿using Microsoft.ML.AutoML;
using Microsoft.ML.AutoML.CodeGen;
using System.Diagnostics;

namespace AutoMLSample.Services;

public class AutoMLMonitor : IMonitor
{
    private static SweepablePipeline _pipeline;
    private readonly List<TrialResult> _bestTrials = new();
    private readonly List<TrialResult> _completedTrials = new();
    private readonly Stopwatch _stopwatch;

    public double? PeakCpu { get; private set; }
    public double? PeakMemoryInMegaByte { get; private set; }

    protected AutoMLMonitor(SweepablePipeline pipeline)
    {
        _pipeline = pipeline;
        _stopwatch = Stopwatch.StartNew();
    }

    public static AutoMLMonitor Create(SweepablePipeline pipeline)
    {
        return new AutoMLMonitor(pipeline);
    }

    public IEnumerable<TrialResult> GetBestTrials() => _bestTrials;
    public IEnumerable<TrialResult> GetCompletedTrials() => _completedTrials;

    public void ReportBestTrial(TrialResult result)
    {
        var trainer = ExtractTrainerName(result.TrialSettings);
        Log.Information(" {@trainer} is BEST trial", trainer.EstimatorType);

        _bestTrials.Add(result);
    }

    public void ReportCompletedTrial(TrialResult result)
    {
        if (result.PeakCpu is not null && (PeakCpu is null || result.PeakCpu > PeakCpu))
        {
            PeakCpu = result.PeakCpu;
        }

        if (result.PeakMemoryInMegaByte is not null && (PeakMemoryInMegaByte is null || result.PeakMemoryInMegaByte > PeakMemoryInMegaByte))
        {
            PeakMemoryInMegaByte = result.PeakMemoryInMegaByte;
        }

        var trainer = ExtractTrainerName(result.TrialSettings);
        Log.Debug(" {@trainer} trial completed in {@timeToTrain} ms", trainer.EstimatorType, result.DurationInMilliseconds);

        _completedTrials.Add(result);
    }

    public void ReportFailTrial(TrialSettings settings, Exception exception = null)
    {
        if (exception.Message.Contains("Operation was canceled."))
        {
            Log.Error(" Trial {@trialId} cancelled. Time budget exceeded.", settings.TrialId);
        }
        else 
        {
            Log.Error(" Trial {@trialId} failed with exception {@message}", settings.TrialId, exception.Message);
        }
    }

    public void ReportRunningTrial(TrialSettings trialSettings)
    {
        // HERE
        // elapsed, remaining, percent progress
        //var trainer = ExtractTrainer(trialSettings);
        //Log.Debug(" {@trainer} trial...", trainer.EstimatorType);
    }

    public string GetBestTrial(TrialResult result)
    {
        _stopwatch.Stop();
        var pipeline = _pipeline.ToString(result.TrialSettings.Parameter);
        return $" {pipeline.ToString().Split("=>").Last()}";
    }

    public static SweepableEstimator ExtractTrainerName(TrialSettings trialSettings)
    {
        trialSettings.Parameter.TryGetValue("_pipeline_", out var pipelineData);
        string pipeline = pipelineData["_SCHEMA_"].AsType<string>();
        var lastEstimator = pipeline.Split("*", StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).Last();
        var trainer = _pipeline.Estimators[lastEstimator];
        return trainer;
    }
}
