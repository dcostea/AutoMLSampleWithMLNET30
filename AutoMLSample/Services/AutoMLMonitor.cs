namespace AutoMLSample.Services;

public class AutoMLMonitor : IMonitor
{
    private readonly SweepablePipeline _pipeline;
    private readonly List<TrialResult> _completedTrials;

    public double? PeakCpu { get; private set; }
    public double? PeakMemoryInMegaByte { get; private set; }

    public AutoMLMonitor(SweepablePipeline pipeline)
    {
        _pipeline = pipeline;
        _completedTrials = new List<TrialResult>();
    }

    public IEnumerable<TrialResult> GetCompletedTrials() => _completedTrials;

    public void ReportBestTrial(TrialResult result)
    {
        var pipeline = _pipeline.ToString(result.TrialSettings.Parameter);
        Log.Information($" Best trial {pipeline.ToString().Split("=>").Last()}");
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

        var timeToTrain = result.DurationInMilliseconds;
        var pipeline = _pipeline.ToString(result.TrialSettings.Parameter);
        Log.Debug($" Completed trial {pipeline.ToString().Split("=>").Last()} in {timeToTrain} ms");
        _completedTrials.Add(result);
    }

    public void ReportFailTrial(TrialSettings settings, Exception exception = null)
    {
        if (exception.Message.Contains("Operation was canceled."))
        {
            Log.Error($" Trial {settings.TrialId} cancelled. Time budget exceeded.");
        }
        else 
        {
            Log.Error($" Trial {settings.TrialId} failed with exception {exception.Message}");
        }
    }

    public void ReportRunningTrial(TrialSettings setting)
    {
        Log.Debug($" Trial {setting.TrialId} is running...");
        return;
    }

    public string GetBestTrial(TrialResult result)
    {
        var pipeline = _pipeline.ToString(result.TrialSettings.Parameter);
        return $" {pipeline.ToString().Split("=>").Last()}";
    }
}
