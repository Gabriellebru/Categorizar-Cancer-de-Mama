// <SnippetUsingsForPaths>
using System;
using System.IO;
using Categorizar_Cancer_de_Mama;
// </SnippetUsingsForPaths>

// <SnippetMLUsings>
using Microsoft.ML;
using Microsoft.ML.Data;
// </SnippetMLUsings>

namespace BreastCancerClustering
{
    class Program
    {
        // <SnippetPaths>
        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "breast-cancer.data");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "BreastCancerModel.zip");

        public static object TesteCancerMama { get; private set; }

        // </SnippetPaths>

        static void Main(string[] args)
        {
            // <SnippetCreateContext>
            var mlContext = new MLContext(seed: 0);
            // </SnippetCreateContext>

            // <SnippetCreateDataView>
            IDataView dataView = mlContext.Data.LoadFromTextFile<BreastCancerData>(_dataPath, hasHeader: false, separatorChar: ',');
            // </SnippetCreateDataView>

            // <SnippetCreatePipeline>
            string featuresColumnName = "Features";
            var pipeline = mlContext.Transforms
                .Concatenate(featuresColumnName, "Class", "age", "menopause", "tumor_size", "inv_nodes", "node_caps", "deg_malig", "breast", "breast_quad", "irradiat")
                .Append(mlContext.Clustering.Trainers.KMeans(featuresColumnName, numberOfClusters: 3));
            // </SnippetCreatePipeline>

            // <SnippetTrainModel>
            var model = pipeline.Fit(dataView);
            // </SnippetTrainModel>

            // <SnippetSaveModel>
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
            {
                mlContext.Model.Save(model, dataView.Schema, fileStream);
            }
            // </SnippetSaveModel>

            // <SnippetPredictor>
            var predictor = mlContext.Model.CreatePredictionEngine<BreastCancerData, ClusterPrediction>(model);
            // </SnippetPredictor>

            // <SnippetPredictionExample>
            var prediction = predictor.Predict(TesteCancerMama.irradiat);
            Console.WriteLine($"Cluster: {prediction.PredictedClusterId}");
            Console.WriteLine($"Distances: {string.Join(" ", prediction.Distances)}");
            // </SnippetPredictionExample>
        }
    }
}
