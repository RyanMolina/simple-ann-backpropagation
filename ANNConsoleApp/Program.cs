using ANNConsoleApp.ArtificialNeuralNetwork;
using System;
using System.Collections.Generic;

namespace ANNConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            const int HIDDENLAYER = 5;
            const int EPOCH = 10000;

            Random random = new Random();

            //AND gate training set
            List<TrainingData> trainingSet = new List<TrainingData>()
            {
                new TrainingData()
                {
                    InputSet = new List<double> {0, 0}, OutputSet = new List<double> {0}
                },
                new TrainingData()
                {
                    InputSet = new List<double> {1, 1}, OutputSet = new List<double> {1}
                },
                new TrainingData()
                {
                    InputSet = new List<double> {1, 0}, OutputSet = new List<double> {0}
                },
                new TrainingData()
                {
                    InputSet = new List<double> {0, 1}, OutputSet = new List<double> {0}
                },

            };


            NeuralNetwork nn = new NeuralNetwork(
                trainingSet[0].InputSet.Count, 
                HIDDENLAYER, 
                trainingSet[0].OutputSet.Count);


            for (int iter = 0; iter < EPOCH; iter++)
            {
                TrainingData trainingData = trainingSet[random.Next(0, trainingSet.Count)];

                List<double> trainingInputs = trainingData.InputSet;
                List<double> trainingOuputs = trainingData.OutputSet;
                nn.BackPropagation(trainingInputs, trainingOuputs);
                Console.WriteLine(iter + ": " + nn.CalculateError(trainingSet));
            }
            List<double> inputs = new List<double>() { 1, 1 };
            List<double> outputs = nn.FeedForward(inputs);

            Console.WriteLine(outputs[0]);
            Console.ReadKey();
        }
    }
}