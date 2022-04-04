using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;


namespace Categorizar_Cancer_de_Mama
{
    internal class BreastCancerData
    {
        [LoadColumn(0)]
        public char Class;

        [LoadColumn(1)]
        public char age;

        [LoadColumn(2)]
        public string menopause;

        [LoadColumn(3)]
        public string tumor_size;
        [LoadColumn(4)]
        public string inv_nodes;

        [LoadColumn(5)]
        public string node_caps;

        [LoadColumn(6)]
        public string deg_malig;

        [LoadColumn(7)]
        public string breast;
        [LoadColumn(8)]
        public string breast_quad;
        [LoadColumn(8)]
        public string irradiat;


    }
    public class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedClusterId;

        [ColumnName("Score")]
        public float[] Distances;
    }
}
