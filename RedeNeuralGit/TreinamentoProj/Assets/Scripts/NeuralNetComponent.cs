using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

namespace NeuralNetUnity
{
    using Perceptron;
    public class NeuralNetComponent : MonoBehaviour
    {
        [SerializeField]
        int m_HiddenCount = 6;
        [SerializeField]
        double m_LearningRate = 0.5;
        [SerializeField]
        double m_Momentum = 0.9;
        [SerializeField]
        double m_MatchTolerance = 0.95;
        [SerializeField]
        int m_EpochCount = 50000;
        [SerializeField]
        int m_TrainingTries = 2;

        [SerializeField]
        UnityEngine.UI.Text m_TextInfo;

        NeuralNet m_NeuralNet;

        List<Vector3> m_Path;
        List<Vector3> m_SmoothPath;
        int m_SmoothCount = GestureData.SmoothCount;

        List<double> m_Data;
        GestureData m_GestureData;

        bool m_IsDrawing;

        bool m_IsAddinggesture;
        int m_CustomGestureCount;

        private void Awake()
        {
            m_Path = new List<Vector3>(m_SmoothCount);
            m_SmoothPath = new List<Vector3>(m_SmoothCount);
            m_Data = new List<double>(m_SmoothCount * 2);

            m_GestureData = new GestureData();

            Debug.Log($"Construindo a rede neural com {m_GestureData.InputCount} entradas, camada intermediária com {m_HiddenCount} neurônios e {m_GestureData.OutputCount} saídas");
            
            m_NeuralNet = new NeuralNet(
                inputCount: m_GestureData.InputCount,
                hiddenCount: m_HiddenCount,
                outputCount: m_GestureData.OutputCount,
                learningRate: m_LearningRate,
                momentum: m_Momentum,
                matchTolerance: m_MatchTolerance);

            m_IsDrawing = false;

            m_IsAddinggesture = false;
            m_CustomGestureCount = 0;

            StartCoroutine(StartTraining());
        }

        IEnumerator StartTraining()
        {
            Debug.Log("===> StartTraining()");

            int trainingCount = 0;
            TrainingResult trainingResult = TrainingResult.Unknown;

            while(trainingResult != TrainingResult.EarlyStopSSE && trainingCount < m_TrainingTries)
            {
                Debug.Log($"Tentativa {trainingCount + 1} de treinamento com época {m_EpochCount}.");
                trainingResult = m_NeuralNet.Train(m_GestureData.TrainingDataInput, m_GestureData.TrainingDataDesiredOutput, m_EpochCount);
                ++trainingCount;
                yield return new WaitForFixedUpdate();
            }
            Debug.Log($"<====== Treinamento encerrado depois de {trainingCount} tentativas.");

            Debug.Log(m_NeuralNet.GetWeightsAsString());
            Debug.Log(m_NeuralNet.GetBiasAsString());
        }

        // Update is called once per frame
        void Update()
        {
            if (Input.GetKeyDown(KeyCode.T))
            {
                StartCoroutine(StartTraining());
                return;
            }

            if (Input.GetKeyDown(KeyCode.L))
            {
                string path = $@"{Environment.GetFolderPath(Environment.SpecialFolder.Desktop)}\NeuralNet.txt";
                Debug.Log($"===> Carregando arquivo: {path}");

                string result = m_NeuralNet.Load(path);

                m_GestureData.UpdateOutputNames(m_NeuralNet.OutputNames);

                Debug.Log(result);
                return;
            }

            if (Input.GetKeyDown(KeyCode.S))
            {
                string path = $@"{Environment.GetFolderPath(Environment.SpecialFolder.Desktop)}\NeuralNet.txt";
                Debug.Log($"===> Salvando arquivo: {path}");

                m_NeuralNet.UpdateOutputNames(m_GestureData.OutputNames);

                string result = m_NeuralNet.Save(path);

                Debug.Log(result);
                return;
            }

            if(Input.GetKeyDown(KeyCode.A) && !m_IsAddinggesture)
            {
                Debug.Log("====> Adicionando novo gesto...");
                m_TextInfo.text = "Adicionando novo gesto...";

                m_IsAddinggesture = true;
            }

            if(!m_IsAddinggesture && !m_NeuralNet.IsTrained)
            {
                return;
            }

            if (Input.GetMouseButtonDown(0))
            {
                StartDrawing();
            }

            if(Input.GetMouseButton(0) && m_IsDrawing)
            {
                m_Path.Add(Input.mousePosition);
            }

            if (Input.GetMouseButtonUp(0))
            {
                StopDrawing();
            }
        }

        void StartDrawing()
        {
            m_Path.Clear();
            m_SmoothPath.Clear();
            m_Data.Clear();

            m_Data.Clear();
        }

        void StopDrawing()
        {
            m_IsDrawing = false;

            if (Smooth())
            {
                CreateData();

                if (m_IsAddinggesture)
                {
                    ++m_CustomGestureCount;
                    m_GestureData.AddGesture($"Custom Gesture #{m_CustomGestureCount}", m_Data.ToArray());
                    Debug.Log($"Novo gesto é Custom Gesture #{m_CustomGestureCount}.");

                    Debug.Log($"Reconstruindo a rede neural com {m_GestureData.InputCount} entradas, camada intermediária com {m_HiddenCount} neurônios e {m_GestureData.OutputCount} saídas.");

                    m_NeuralNet = new NeuralNet(
                        inputCount: m_GestureData.InputCount,
                        hiddenCount: m_HiddenCount,
                        outputCount: m_GestureData.OutputCount,
                        learningRate: m_LearningRate,
                        momentum: m_Momentum,
                        matchTolerance: m_MatchTolerance);

                    m_IsAddinggesture = false;
                    Debug.Log($"Novo gesto adicionado: Custom Gesture #{m_CustomGestureCount}.");
                    m_TextInfo.text = $"Novo gesti adicionado: Custom Gesture #{m_CustomGestureCount}";

                    StartCoroutine(StartTraining());

                    return;
                }

                (int index, double value) result = m_NeuralNet.GetHighestOutput(m_Data.ToArray());

                if(result.value >= m_MatchTolerance)
                {
                    Debug.Log($">>>>> Melhor resultado é: neurônio de saída {result.index} com valor {result.value}\nGESTO DETECTADO: {m_GestureData.GetValueAsString(result.index).ToUpper()}."); 
                    m_TextInfo.text = $"Detectado: {m_GestureData.GetValueAsString(result.index).ToUpper()}";
                }
                else
                {
                    Debug.Log($">>>>> Melhor resultado é: neurônio de saída {result.index} com valor {result.value}\nPossível gesto detectado: {m_GestureData.GetValueAsString(result.index)}.");
                    m_TextInfo.text = $"Possivelmente detectado: {m_GestureData.GetValueAsString(result.index)}";
                }
            }
        }

        bool Smooth()
        {
            int pathCount = m_Path.Count;
            if(pathCount < m_SmoothCount)
            {
                Debug.LogWarning($"Não há dados suficientes para reconhecer o gesto ({pathCount}/{m_SmoothCount}).");
                return false;
            }

            for(int i = 0; i < pathCount; ++i)
            {
                m_SmoothPath.Add(m_Path[i]);
            }

            while(m_SmoothPath.Count > m_SmoothCount)
            {
                double shortestLength = 99999999;

                int pointMarker = 0;

                for(int spanFront = 2; spanFront < m_SmoothPath.Count - 1; ++spanFront)
                {
                    double length = Math.Sqrt(
                        (m_SmoothPath[spanFront - 1].x - m_SmoothPath[spanFront].x) *
                        (m_SmoothPath[spanFront - 1].x - m_SmoothPath[spanFront].x) +
                        (m_SmoothPath[spanFront - 1].y - m_SmoothPath[spanFront].y) *
                        (m_SmoothPath[spanFront - 1].y - m_SmoothPath[spanFront].y));

                    if(length < shortestLength)
                    {
                        shortestLength = length;
                        pointMarker = spanFront;
                    }
                }

                float newX = (m_SmoothPath[pointMarker - 1].x + m_SmoothPath[pointMarker].x * 0.5f);
                float newY = (m_SmoothPath[pointMarker - 1].y + m_SmoothPath[pointMarker].y * 0.5f);
                m_SmoothPath[pointMarker - 1] = new Vector3(newX, newY);
                m_SmoothPath.RemoveAt(pointMarker);
            }

            return true;
        }

        void CreateData()
        {
            float x, y;
            for(int i = 1; i < m_SmoothPath.Count; ++i)
            {
                x = m_SmoothPath[i].x - m_SmoothPath[i - 1].x;
                y = m_SmoothPath[i].y - m_SmoothPath[i - 1].y;

                Vector3 v = new Vector3(x, y);
                v.Normalize();

                m_Data.Add(v.x);
                m_Data.Add(v.y);
            }
        }
    }
}
