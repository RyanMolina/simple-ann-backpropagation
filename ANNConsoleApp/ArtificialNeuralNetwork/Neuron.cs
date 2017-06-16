using System;
using System.Collections.Generic;
using System.Text;

namespace ANNConsoleApp.ArtificialNeuralNetwork
{
    public class Neuron
    {
        public double Bias { get; set; }

        public List<double> Weights { get; set; }

        public List<double> Inputs { get; set; }

        public double Output { get; set; }


        public Neuron(double bias)
        {
            Bias = bias;
            Weights = new List<double>();
        }


        public double CalculateOutput(List<double> inputs)
        {
            Inputs = inputs;
            Output = Squash(CalculateNetInput());
            return Output;
        }


        public double CalculateNetInput()
        {
            double total = 0;
            for (int i = 0; i < Inputs.Count; i++ )
            {
                total += Inputs[i] * Weights[i];
            }
            return total + Bias;
        }


        public double Squash(double totalNetInput)
        {
            return 1 / (1 + Math.Exp(-totalNetInput));
        }


        public double CalculatePdErrorWrtNetInput(double targetOutput)
        {
            return CalculatePdErrorWrtOutput(targetOutput) * 
                CalculatePdNetInputWrtInput();
        }


        public double CalculateError(double targetOutput)
        {
            return 0.5 * Math.Pow(targetOutput - Output, 2);
        }


        public double CalculatePdErrorWrtOutput(double targetOutput)
        {
            return -(targetOutput - Output);
        }


        public double CalculatePdNetInputWrtInput()
        {
            return Output * (1 - Output);
        }

        public double CalculatePdNetInputWrtWeight(int index)
        {
            return Inputs[index];
        }
    }

}
