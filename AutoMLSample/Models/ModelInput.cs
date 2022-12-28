namespace AutoMLSample.Models;

internal class ModelInput
{
    [LoadColumn(0)]
    [ColumnName("Survived")]
    public bool Label { get; set; }

    [LoadColumn(1)]
    [ColumnName("Pclass")]
    public float Pclass { get; set; }

    [LoadColumn(2)]
    [ColumnName("Sex")]
    public string Sex { get; set; }

    [LoadColumn(3)]
    [ColumnName("Age")]
    public float Age { get; set; }

    [LoadColumn(4)]
    [ColumnName("SibSp")]
    public float SibSp { get; set; }

    [LoadColumn(5)]
    [ColumnName("Parch")]
    public float Parch { get; set; }

    [LoadColumn(6)]
    [ColumnName("Fare")]
    public float Fare { get; set; }
}

//internal class ModelInput
//{
//    [LoadColumn(0)]
//    public float Temperature { get; set; }

//    [LoadColumn(1)]
//    public float Temperature2 { get; set; }

//    [LoadColumn(2)]
//    public float Luminosity { get; set; }

//    [LoadColumn(3)]
//    public float Infrared { get; set; }

//    [LoadColumn(4)]
//    public float Distance { get; set; }

//    [LoadColumn(5)]
//    public float PIR { get; set; }

//    [LoadColumn(6)]
//    public float Humidity { get; set; }

//    ////[LoadColumn(7)]
//    ////public DateTime CreatedAt { get; set; }

//    [ColumnName("Source"), LoadColumn(8)]
//    public string Label { get; set; }
//}
