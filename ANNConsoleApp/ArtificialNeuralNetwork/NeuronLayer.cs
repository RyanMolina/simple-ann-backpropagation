using System;
using System.Collections.Generic;
using System.Text;

namespace ANNConsoleApp.ArtificialNeuralNetwork
{
    public class NeuronLayer
    {

        public double Bias { get; set; }

        public List<Neuron> Neurons { get; set; }


        public NeuronLayer(int neuronCount, double bias)
        {
            Bias = bias;
            Neurons = new List<Neuron>();

            for(int i = 0; i < neuronCount; i++)
            {
                Neurons.Add(new Neuron(Bias));
            }
        }

        public List<double> FeedForward(List<double> inputs)
        {
            List<double> outputs = new List<double>();
            foreach(Neuron neuron in Neurons)
            {
                outputs.Add(neuron.CalculateOutput(inputs));
            }
            return outputs;
        }


        public List<double> GetOutputs()
        {
            List<double> outputs = new List<double>();

            foreach(Neuron neuron in Neurons)
            {
                outputs.Add(neuron.Output);
            }
            return outputs;
        }

    }
}
