namespace AutoMLSample.Models;

//internal class ModelOutput
//{
//    public string PredictedLabel;

//    //[ColumnName("Score")]
//    public float[] Score;
//}

#region model output class

public class ModelOutput
{
   
    [ColumnName(@"Features")]
    public float[] Features { get; set; }

    [ColumnName(@"PredictedLabel")]
    public string PredictedLabel { get; set; }

    [ColumnName(@"Score")]
    public float[] Score { get; set; }

}

#endregion