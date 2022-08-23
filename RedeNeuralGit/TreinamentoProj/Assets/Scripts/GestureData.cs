using System.Collections;
using System;
using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetUnity
{
    public class GestureData
    {
        static double[][] FixedTrainingDataInput =
        {
            new double[]//Direita
            {
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
                1.0, 0.0,
            },
            new double[]//Esquerda
            {
                -1.0, 0.0,
                -1.0, 0.0,
                -1.0, 0.0,
                -1.0, 0.0,
                -1.0, 0.0,
                -1.0, 0.0,
                -1.0, 0.0,
                -1.0, 0.0,
                -1.0, 0.0,
                -1.0, 0.0,
                -1.0, 0.0,
                -1.0, 0.0,
            },
            new double[] //Cima (seria baixo, mas o eixo da Unity é diferente)
            {
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
                0.0, 1.0,
            },
            new double[] //Baixo (seria cima, mas o eixo da Unity é diferente)
            {
                0.0, -1.0,
                0.0, -1.0,
                0.0, -1.0,
                0.0, -1.0,
                0.0, -1.0,
                0.0, -1.0,
                0.0, -1.0,
                0.0, -1.0,
                0.0, -1.0,
                0.0, -1.0,
                0.0, -1.0,
                0.0, -1.0,
            },
        };
        static string[] FixedOutputNames =
        {
            "Direita",
            "Esquerda",
            "Cima",
            "Baixo"
        };

        List<double[]> m_TrainingDataInput;
        List<double[]> m_TrainingDataDesiredOutput;
        List<string> m_OutputNames;

        public static int SmoothCount = FixedTrainingDataInput[0].Length / 2 + 1;

        public double[][] TrainingDataInput { get => m_TrainingDataInput.ToArray(); }
        public double[][] TrainingDataDesiredOutput { get => m_TrainingDataDesiredOutput.ToArray(); }
        public int InputCount { get => m_TrainingDataInput[0].Length; }
        public int OutputCount { get => m_TrainingDataInput.Count; }
        public string GetValueAsString(int value) => m_OutputNames[value];
        public string[] OutputNames { get => m_OutputNames.ToArray(); }

        public GestureData()
        {
            int classCount = FixedTrainingDataInput.Length;
            m_TrainingDataInput = new List<double[]>(classCount);
            m_TrainingDataDesiredOutput = new List<double[]>(classCount);
            m_OutputNames = new List<string>(classCount);

            for(int i = 0; i< classCount; ++i)
            {
                m_TrainingDataInput.Add(FixedTrainingDataInput[i]);
                m_OutputNames.Add(FixedOutputNames[i]);
            }

            UpdateDesiredOutput();
        }

        void UpdateDesiredOutput()
        {
            m_TrainingDataDesiredOutput.Clear();

            int classCount = m_TrainingDataInput.Count;
            for(int i = 0; i< classCount; ++i)
            {
                double[] desiredOutput = new double[classCount];
                for(int j =0; j < classCount; ++j)
                {
                    desiredOutput[j] = 0.0;
                }
                desiredOutput[i] = 1.0;
                m_TrainingDataDesiredOutput.Add(desiredOutput);
            }
        }

        public void UpdateOutputNames(String[] names)
        {
            m_OutputNames.Clear();
            for(int i = 0; i < names.Length; ++i)
            {
                m_OutputNames.Add(names[i]);
            }
        }

        public void AddGesture(string name, double[] data)
        {
            m_TrainingDataInput.Add(data);
            m_OutputNames.Add(name);

            UpdateDesiredOutput();
        }
    }
}