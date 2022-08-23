using System.Collections;
using System;
using System.Text;
using System.Collections.Generic;
using UnityEngine;

namespace Perceptron
{
    public class Neuron
    {
        double[] m_Inputs;
        double[] m_Weights;
        public double[] Weights { get => m_Weights; }
        double m_Bias;
        public double Bias { get => m_Bias; }
        double m_WeightedSum;
        double m_Output;
        public double Output { get => m_Output; }

        //Momentum.
        double[] m_PreviousWeights;
        double m_PreviousBias;
        double m_Momentum;

        Func<double, double> ActivationFunction;

        public static int StepFunction(double x) => x >= 0 ? 1 : 0;
        public static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        public static double SigmoidDerivative(double x) => x * (1.0 - x);

        public static double DotProduct(double[] a, double[] b)
        {
            int length = a.Length;
            double result = 0.0;

            for(int i = 0; i < length; ++i)
            {
                result += a[i] * b[i];
            }

            return result;
        }

        public Neuron(
            double[] inputs,
            double[] weights,
            double bias,
            double momentum,
            Func<double, double> activationFunction = null)
        {
            int inputCount = inputs.Length;

            m_Inputs = new double[inputCount];
            m_Weights = new double[inputCount];
            m_PreviousWeights = new double[inputCount];
            for(int i = 0; i < inputCount; ++i)
            {
                m_Inputs[i] = inputs[i];
                m_Weights[i] = weights[i];
                m_PreviousWeights[i] = 0.0;
            }

            m_Bias = bias;
            m_PreviousBias = 0.0;
            m_Momentum = momentum;

            ActivationFunction = activationFunction ?? Sigmoid;

            UpdateWeightedSum();
            UpdateOutput();
        }

        void UpdateWeightedSum()
        {
            m_WeightedSum = m_Bias + DotProduct(m_Inputs, m_Weights);
        }

        void UpdateOutput()
        {
            m_Output = ActivationFunction(m_WeightedSum);
        }

        public void AdjustWeights(double learningRate, double error)
        {
            //Ajuste de pesos com momentum.
            double weightUpdate = learningRate * error;
            m_Bias += weightUpdate + m_PreviousBias * m_Momentum;
            m_PreviousBias = weightUpdate;

            int weightCount = m_Weights.Length;
            for(int i = 0; i < weightCount; ++i)
            {
                weightUpdate = learningRate * error * m_Inputs[i];
                m_Weights[i] += weightUpdate + m_PreviousWeights[i] * m_Momentum;
                m_PreviousWeights[i] = weightUpdate;
            }
        }

        public override string ToString()
        {
            int inputCount = m_Inputs.Length;

            StringBuilder sb = new StringBuilder();
            sb.Append("Inputs: ");
            for(int i = 0; i < inputCount; ++i)
            {
                sb.Append($"{m_Inputs[i]:F5};");
            }
            sb.Append("\nWeights: ");
            for(int i =0; i < inputCount; i++)
            {
                sb.Append($"{m_Weights[i]:F5};");
            }
            sb.Append($"\nBias: {m_Bias:F5}");
            sb.Append($"\nWeightedSum: {m_WeightedSum:F5}");
            sb.Append($"\nOuput: {Output:F5}");

            return sb.ToString();
        }
    }
}