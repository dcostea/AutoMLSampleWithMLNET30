namespace AutoMLSample.Helpers;

public class MulticlassExperimentProgressHandler : IProgress<RunDetail<MulticlassClassificationMetrics>>
{
    private int _iterationIndex;

    public void Report(RunDetail<MulticlassClassificationMetrics> iterationResult)
    {
        if (_iterationIndex++ == 0)
        {
            ConsoleHelpers.PrintMulticlassClassificationMetricsHeader();
        }

        if (iterationResult.Exception != null)
        {
            ConsoleHelpers.PrintIterationException(iterationResult.Exception, iterationResult.TrainerName.Replace("Multi", ""));
        }
        else
        {
            ConsoleHelpers.PrintIterationMetrics(_iterationIndex, iterationResult.TrainerName.Replace("Multi", ""),
                iterationResult.ValidationMetrics, iterationResult.RuntimeInSeconds);
        }
    }
}
