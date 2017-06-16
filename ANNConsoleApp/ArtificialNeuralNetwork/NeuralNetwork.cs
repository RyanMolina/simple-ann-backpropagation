using System;
using System.Collections.Generic;
using System.Text;

namespace ANNConsoleApp.ArtificialNeuralNetwork
{
    public class NeuralNetwork
    {
        public int InputLayerCount { get; set; }

        public int HiddenLayerCount { get; set; }

        public int OutputLayerCount { get; set; }

        public NeuronLayer HiddenLayer { get; set; }

        public NeuronLayer OutputLayer { get; set; }

        private double learningRate;

        public double LearningRate
        {
            get { return learningRate; }
            set
            {
                if (value > 0 && value <= 1)
                {
                    this.learningRate = value;
                }
                else
                {
                    throw new ArgumentOutOfRangeException("Invalid value. Value must be between 0 and 1");
                }
            }
        }



        public NeuralNetwork(int inputLayerCount, int hiddenLayerCount, int outputLayerCount) : 
                    this (inputLayerCount, hiddenLayerCount, outputLayerCount, null, 0, null, 0)
        {
            // Constructor chaining
        }

        public NeuralNetwork(
                int inputLayerCount, 
                int hiddenLayerCount, 
                int outputLayerCount,
                List<double> hiddenLayerWeights,
                double hiddenLayerBias,
                List<double> outputLayerWeights,
                double outputLayerBias)
        {
            LearningRate = 0.9;

            InputLayerCount = inputLayerCount;

            HiddenLayer = new NeuronLayer(hiddenLayerCount, hiddenLayerBias);
            OutputLayer = new NeuronLayer(outputLayerCount, outputLayerBias);

            if(hiddenLayerWeights != null)
            {
                InitWeightsInputToHiddenLayerNeurons(hiddenLayerWeights);
            } 
            else
            {
                InitWeightsInputToHiddenLayerNeurons();
            }


            if(outputLayerWeights != null)
            {
                InitWeightsHiddenLayerToOutputLayerNeurons(outputLayerWeights);
            }
            else
            {
                InitWeightsHiddenLayerToOutputLayerNeurons();

            }

        }


        public void InitWeightsInputToHiddenLayerNeurons()
        {
            int weightCount = 0;
            Random random = new Random();
            for (int i = 0; i < HiddenLayer.Neurons.Count; i++)
            {
                for (int j = 0; j < InputLayerCount; j++)
                {
                    HiddenLayer.Neurons[i].Weights.Add(random.NextDouble());
                    weightCount++;
                }
            }
        }


        public void InitWeightsInputToHiddenLayerNeurons(List<double> hiddenLayerWeights)
        {
            int weightCount = 0;
            for(int i = 0; i < HiddenLayer.Neurons.Count; i++)
            {
                for(int j = 0; j < InputLayerCount; j++)
                {
                    HiddenLayer.Neurons[i].Weights.Add(hiddenLayerWeights[weightCount]);
                    weightCount++;
                }
            }
        }


        public void InitWeightsHiddenLayerToOutputLayerNeurons()
        {
            int weightCount = 0;
            Random random = new Random();
            for(int i = 0; i < OutputLayer.Neurons.Count; i++)
            {
                for(int j = 0; j < HiddenLayer.Neurons.Count; j++)
                {
                    OutputLayer.Neurons[i].Weights.Add(random.NextDouble());
                    weightCount++;
                }   
            }
        }


        public void InitWeightsHiddenLayerToOutputLayerNeurons(List<double> outputLayerWeights)
        {
            int weightCount = 0;
            for(int i = 0; i < OutputLayer.Neurons.Count; i++)
            {
                for(int j = 0; j < HiddenLayer.Neurons.Count; j++)
                {
                    OutputLayer.Neurons[i].Weights.Add(outputLayerWeights[i]);
                    weightCount++;
                }
            }
        }


        public List<double> FeedForward(List<double> inputs)
        {
            List<double> hiddenLayerOutputs = HiddenLayer.FeedForward(inputs);
            return OutputLayer.FeedForward(hiddenLayerOutputs);
        }


        public void BackPropagation(List<double> inputs, List<double> outputs)
        {
            FeedForward(inputs);


            // 1. Output neuron deltas
            double[] pdErrorsWrtOutputNeuronNetInput = new double[OutputLayer.Neurons.Count];
            for(int i = 0; i < OutputLayer.Neurons.Count; i++)
            {
                pdErrorsWrtOutputNeuronNetInput[i] = 
                        OutputLayer.Neurons[i].CalculatePdErrorWrtNetInput(outputs[i]);

            }
            // 2. Hidden neuron deltas
            double[] pdErrorsWrtHiddenNeuronInput = new double[HiddenLayer.Neurons.Count];
            for(int i = 0; i < HiddenLayer.Neurons.Count; i++)
            {
                double dErrorWrtHiddenNeuronOuput = 0;
                for(int j = 0; j < OutputLayer.Neurons.Count; j++)
                {
                    dErrorWrtHiddenNeuronOuput += pdErrorsWrtOutputNeuronNetInput[j] * OutputLayer.Neurons[j].Weights[i];
                }
                pdErrorsWrtHiddenNeuronInput[i] = dErrorWrtHiddenNeuronOuput * HiddenLayer.Neurons[i].CalculatePdNetInputWrtInput();
            }

            // 3. Update output neuron weights
            for(int i = 0; i < OutputLayer.Neurons.Count; i++)
            {
                for(int j = 0; j < OutputLayer.Neurons[i].Weights.Count; j++)
                {
                    double pdErrorWrtWeight = pdErrorsWrtOutputNeuronNetInput[i] * OutputLayer.Neurons[i].CalculatePdNetInputWrtWeight(j);
                    OutputLayer.Neurons[i].Weights[j] -= LearningRate * pdErrorWrtWeight;
                }
            }


            // 4. Update hidden neuron weights
            for(int i = 0; i < HiddenLayer.Neurons.Count; i++)
            {
                for(int j = 0; j < HiddenLayer.Neurons[i].Weights.Count; j++)
                {
                    double pdErrorWrtWeight = pdErrorsWrtHiddenNeuronInput[i] * HiddenLayer.Neurons[i].CalculatePdNetInputWrtWeight(j);
                    HiddenLayer.Neurons[i].Weights[j] -= LearningRate * pdErrorWrtWeight;
                }
            }

        }


        public double CalculateError(List<TrainingData> trainingSet)
        {
            double error = 0;

            foreach(TrainingData trainingData in trainingSet)
            {
                List<double> trainingInputs = trainingData.InputSet;
                List<double> trainingOuputs = trainingData.OutputSet;

                FeedForward(trainingInputs);

                for(int i = 0; i < OutputLayer.Neurons.Count; i++)
                {
                    error += OutputLayer.Neurons[i].CalculateError(trainingOuputs[i]);
                }
            }
            return error;
        }

    }
}
