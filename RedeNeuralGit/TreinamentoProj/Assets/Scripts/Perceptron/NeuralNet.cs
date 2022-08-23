using System;
using System.Globalization;
using System.IO;
using System.Text;

namespace Perceptron
{
    public enum TrainingResult
    {
        FinalEpoch,
        EarlyStopSSE,
        ErrorInputSizeMismatch,
        Unknown,
    };
    public class NeuralNet
    {
        int m_InputCount;
        int m_HiddenCount;
        int m_OutputCount;

        double m_LearningRate;
        double m_Momentum;
        double m_MatchTolerance;

        double[][] m_WeightsInputToHidden;
        double[] m_BiasInputToHidden;
        double[][] m_WeightsHiddenToOutput;
        double[] m_BiasHiddenToOutput;

        Neuron[] m_HiddenLayer;
        Neuron[] m_OutputLayer;

        bool m_IsTrained;
        public bool IsTrained { get => m_IsTrained; }

        Func<double, double> ActivationFunction;
        Func<double, double> DerivativeFunction;

        string[] m_OutputNames;
        public string[] OutputNames { get => m_OutputNames; }

        public NeuralNet(
            int inputCount,
            int hiddenCount, 
            int outputCount,
            double learningRate = 0.1,
            double momentum = 0.9,
            double matchTolerance = 0.95,
            Func<double, double> activationFunction = null,
            Func<double, double> derivativeFunction = null)
        {
            Random random = new Random((int)DateTime.Now.Ticks);
            Func<double, double, double> RandomBetween = (min, max) => random.NextDouble() * (max - min) + min;

            m_InputCount = inputCount;
            m_HiddenCount = hiddenCount;
            m_OutputCount = outputCount;

            m_WeightsInputToHidden = new double[m_HiddenCount][];
            m_BiasInputToHidden = new double[m_HiddenCount];
            m_HiddenLayer = new Neuron[m_HiddenCount];
            for(int i = 0; i < m_HiddenCount; i++)
            {
                m_BiasInputToHidden[i] = 1.0;

                m_WeightsInputToHidden[i] = new double[m_InputCount];
                for(int j = 0; j < m_InputCount; ++j)
                {
                    m_WeightsInputToHidden[i][j] = RandomBetween(-1.0, 1.0);
                }
            }

            m_WeightsHiddenToOutput = new double[m_OutputCount][];
            m_BiasHiddenToOutput = new double[m_OutputCount];
            m_OutputLayer = new Neuron[m_OutputCount];
            m_OutputNames = new string[m_OutputCount];

            for(int i = 0; i < m_OutputCount; ++i)
            {
                m_BiasHiddenToOutput[i] = 1.0;

                m_WeightsHiddenToOutput[i] = new double[m_HiddenCount];
                for(int j =0; j < m_HiddenCount; ++j)
                {
                    m_WeightsHiddenToOutput[i][j] = RandomBetween(-1.0, 1.0);
                }
            }

            m_LearningRate = learningRate;
            m_Momentum = momentum;
            m_MatchTolerance = matchTolerance;

            ActivationFunction = activationFunction ?? Neuron.Sigmoid;
            DerivativeFunction = derivativeFunction ?? Neuron.SigmoidDerivative;

            m_IsTrained = false;
        }

        public TrainingResult Train(double[][] input, double[][] desiredOutput, int epochCount, double sseThrehold = 0.003)
        {
            if(input[0].Length != m_InputCount)
            {
                return TrainingResult.ErrorInputSizeMismatch;
            }

            int traingCount = input.Length;
            double sse;
            double error;
            double errorOutput;
            double outputLayerSumError;
            double[] hiddenLayerOutput = new double[m_HiddenCount];
            double[] derivativesHiddenLayer = new double[m_HiddenCount];
            double[] derivativesOutputLayer = new double[m_OutputCount];

            for(int epoch = 0; epoch < epochCount; ++epoch)
            {
                sse = 0.0;

                for(int trainingIndex = 0; trainingIndex < traingCount; ++trainingIndex)
                {
                    // (FeedForward)
                    for(int i = 0; i < m_HiddenCount; ++i)
                    {
                        m_HiddenLayer[i] = new Neuron(input[trainingIndex], m_WeightsInputToHidden[i], m_BiasInputToHidden[i], m_Momentum, ActivationFunction);
                        hiddenLayerOutput[i] = m_HiddenLayer[i].Output;
                    }
                    for(int i = 0; i < m_OutputCount; ++i)
                    {
                        m_OutputLayer[i] = new Neuron(hiddenLayerOutput, m_WeightsHiddenToOutput[i], m_BiasHiddenToOutput[i], m_Momentum, ActivationFunction);
                    }

                    //(Backpropagatin)
                    //4. Para cada neurônio da camada de saída:
                    outputLayerSumError = 0.0;
                    for(int i = 0; i < m_OutputCount; ++i)
                    {
                        // saída obtida
                        derivativesOutputLayer[i] = DerivativeFunction(m_OutputLayer[i].Output);

                        // erro = (saída esperada - saída obtida)
                        errorOutput = desiredOutput[trainingIndex][i] - m_OutputLayer[i].Output;

                        //4a. Calculamos o erro usando a derivada da função de ativação
                        // (erro = (saída esperada - saída obtida) * (saída obtida)
                        error = derivativesOutputLayer[i] * errorOutput;

                        //4b Ajustamos os pesos da camada de saída com um learning rate alfa e o erro calculado no passo anterior
                        m_OutputLayer[i].AdjustWeights(m_LearningRate, error);
                        m_BiasHiddenToOutput[i] = m_OutputLayer[i].Bias;
                        for (int j =0; j < m_HiddenCount; ++j)
                        {
                            m_WeightsHiddenToOutput[i][j] = m_OutputLayer[i].Weights[j];

                            //4c Salvamos o somatório dos erros multiplicados pelos pesos
                            //valor que será usado para ajustar os pesos das camadas intermediárias
                            outputLayerSumError += error * m_OutputLayer[i].Weights[j];
                        }
                        outputLayerSumError += error * m_OutputLayer[i].Bias;

                        //4d Atualizamos o erro total da rede com o erro quadrático atual
                        // (erroTotal = erroTotal + (saída esperada - saída obtida) ^2
                        sse += (errorOutput * errorOutput);
                    }

                    //5 Para cada neurônio das camadas intermediárias:
                    for(int i =0; i < m_HiddenCount; i++)
                    {
                        derivativesHiddenLayer[i] = DerivativeFunction(hiddenLayerOutput[i]);

                        error = derivativesHiddenLayer[i] * outputLayerSumError;

                        m_HiddenLayer[i].AdjustWeights(m_LearningRate, error);
                        m_BiasInputToHidden[i] = m_HiddenLayer[i].Bias;
                        for(int j = 0; j < m_InputCount; ++j)
                        {
                            m_WeightsInputToHidden[i][j] = m_HiddenLayer[i].Weights[j];
                        }
                    }
                }

                if(sse < sseThrehold)
                {
                    Console.WriteLine($">>>>>> Treinamento concluído na época {epoch}!\n>>>>>> A soma dos erros ao quadrado está abaixo do limite");
                    m_IsTrained = true;

                    return TrainingResult.EarlyStopSSE;
                }
            }

            m_IsTrained = true;

            return TrainingResult.FinalEpoch;
        }

        public double[] CalculateOutput(double[] data)
        {
            double[] hiddenLayerOutput = new double[m_HiddenCount];
            double[] output = new double[m_OutputCount];
            for(int i = 0; i < m_OutputCount; i++)
            {
                m_HiddenLayer[i] = new Neuron(data, m_WeightsInputToHidden[i], m_BiasInputToHidden[i], m_Momentum, null);
                hiddenLayerOutput[i] = m_HiddenLayer[i].Output;
            }
            for(int i =0; i < m_OutputCount; ++i)
            {
                m_OutputLayer[i] = new Neuron(hiddenLayerOutput, m_WeightsHiddenToOutput[i], m_BiasHiddenToOutput[i], m_Momentum, null);
                output[i] = m_OutputLayer[i].Output;
            }

            return output;
        }

        public ValueTuple<int , double> GetHighestOutput(double[] data)
        {
            double[] result = CalculateOutput(data);

            int highest = 0;
            for(int i = 1; i < result.Length; ++i)
            {
                if(result[i] > result[highest])
                {
                    highest = i;
                }
            }

            return (highest, result[highest]);
        }

        public void UpdateOutputNames(string[] names)
        {
            for(int i = 0; i < m_OutputCount; i++)
            {
                m_OutputNames[i] = names[i];
            }
        }
        public string GetWeightsAsString()
        {
            StringBuilder sb = new StringBuilder();

            for(int i = 0; i < m_HiddenCount; ++i)
            {
                sb.Append($"in´put_to_hidden[{i}]: ");
                for(int j = 0; j < m_InputCount; ++j)
                {
                    sb.Append($"{m_WeightsInputToHidden[i][j]};");
                }
                sb.Append('\n');
            }
            for(int i =0; i < m_OutputCount; ++i)
            {
                sb.Append($"hidden_to_output[{i}]: ");
                for(int j = 0; j < m_HiddenCount; ++j)
                {
                    sb.Append($"{m_WeightsHiddenToOutput[i][j]};");
                }
                sb.Append('\n');
            }

            return sb.ToString();
        }

        public string GetBiasAsString()
        {
            StringBuilder sb = new StringBuilder();

            sb.Append($"bias_input_to_hidden: ");
            for(int i = 0; i < m_HiddenCount; ++i)
            {
                sb.Append($"{m_BiasInputToHidden[i]};");
            }
            sb.Append("\nbias_hidden_to_output: ");
            for(int i = 0; i < m_OutputCount; ++i)
            {
                sb.Append($"{m_BiasHiddenToOutput[i]};");
            }

            return sb.ToString();
        }

        public string Load(string filename)
        {
            CultureInfo cultureInfo = CultureInfo.CurrentCulture.Clone() as CultureInfo;
            string originalNumberDecimalSeparator = cultureInfo.NumberFormat.NumberDecimalSeparator;
            cultureInfo.NumberFormat.NumberDecimalSeparator = ".";
            CultureInfo.CurrentCulture = cultureInfo;

            try
            {
                StreamReader sr = new StreamReader(filename);
                if (sr != null)
                {
                    string[] separatorKeyValue = { "=" };
                    string[] separatorData = { ";" };

                    string line;
                    string[] lineContents;
                    int initialConfig = 7;
                    while (initialConfig > 0)
                    {
                        line = sr.ReadLine();
                        lineContents = line.Split(separatorKeyValue, StringSplitOptions.None);

                        switch (lineContents[0].ToUpper())
                        {
                            case "INPUTCOUNT":
                                m_InputCount = int.Parse(lineContents[1]);
                                break;
                            case "HIDDENCOUNT":
                                m_HiddenCount = int.Parse(lineContents[1]);
                                break;
                            case "OUTPUTCOUNT":
                                m_OutputCount = int.Parse(lineContents[1]);
                                break;
                            case "LEARNINGRATE":
                                m_LearningRate = double.Parse(lineContents[1]);
                                break;
                            case "MOMENTUM":
                                m_Momentum = double.Parse(lineContents[1]);
                                break;
                            case "MATCHTOLERANCE":
                                m_MatchTolerance = double.Parse(lineContents[1]);
                                break;
                            case "ISTRAINED":
                                m_IsTrained = bool.Parse(lineContents[1]);
                                break;
                        }

                        --initialConfig;
                    }

                    m_WeightsInputToHidden = new double[m_HiddenCount][];
                    m_BiasInputToHidden = new double[m_HiddenCount];
                    m_HiddenLayer = new Neuron[m_HiddenCount];

                    for (int i = 0; i < m_HiddenCount; i++)
                    {
                        line = sr.ReadLine();
                        lineContents = line.Split(separatorData, StringSplitOptions.None);

                        m_WeightsInputToHidden[i] = new double[m_InputCount];
                        for (int j = 0; j < m_InputCount; ++j)
                        {
                            m_WeightsInputToHidden[i][j] = double.Parse(lineContents[j]);
                        }
                    }

                    line = sr.ReadLine();
                    lineContents = line.Split(separatorData, StringSplitOptions.None);
                    for (int i = 0; i < m_HiddenCount; ++i)
                    {
                        m_BiasInputToHidden[i] = double.Parse(lineContents[i]);
                    }

                    m_WeightsHiddenToOutput = new double[m_OutputCount][];
                    m_BiasHiddenToOutput = new double[m_OutputCount];
                    m_OutputLayer = new Neuron[m_OutputCount];
                    m_OutputNames = new string[m_OutputCount];

                    for (int i = 0; i < m_OutputCount; ++i)
                    {
                        line = sr.ReadLine();
                        lineContents = line.Split(separatorData, StringSplitOptions.None);

                        m_WeightsHiddenToOutput[i] = new double[m_HiddenCount];
                        for (int j = 0; j < m_HiddenCount; j++)
                        {
                            m_WeightsHiddenToOutput[i][j] = double.Parse(lineContents[j]);
                        }
                    }

                    line = sr.ReadLine();
                    lineContents = line.Split(separatorData, StringSplitOptions.None);
                    for (int i = 0; i < m_OutputCount; i++)
                    {
                        m_BiasHiddenToOutput[i] = double.Parse(lineContents[i]);
                    }

                    line = sr.ReadLine();
                    lineContents = line.Split(separatorData, StringSplitOptions.None);
                    UpdateOutputNames(lineContents);

                    sr.Close();
                }
            }
            catch (Exception e)
            {
                cultureInfo.NumberFormat.NumberDecimalSeparator = originalNumberDecimalSeparator;
                return $"*** ERROR! ***\n{e}";
            }
            cultureInfo.NumberFormat.NumberDecimalSeparator = originalNumberDecimalSeparator;
            return "OK!";
        }

        public string Save(string filename)
        {
            CultureInfo cultureInfo = CultureInfo.CurrentCulture.Clone() as CultureInfo;
            string originalNumberDecimalSeparator = cultureInfo.NumberFormat.NumberDecimalSeparator;
            cultureInfo.NumberFormat.NumberDecimalSeparator = ".";
            CultureInfo.CurrentCulture = cultureInfo;

            try
            {
                StreamWriter sw = new StreamWriter(filename);
                if(sw != null)
                {
                    StringBuilder sb = new StringBuilder();
                    sb.Append($"InputCount={m_InputCount}\n");
                    sb.Append($"HiddenCount={m_HiddenCount}\n");
                    sb.Append($"OutputCount={m_OutputCount}\n");
                    sb.Append($"LearningRate={m_LearningRate}\n");
                    sb.Append($"Momentum={m_Momentum}\n");
                    sb.Append($"MatchTolerance={m_MatchTolerance}\n");
                    sb.Append($"IsTrained={(m_IsTrained ? "true":"false")}\n");
                    for(int i = 0; i < m_HiddenCount; i++)
                    {
                        for(int j =0; j < m_InputCount; ++j)
                        {
                            sb.Append($"{m_WeightsInputToHidden[i][j]};");
                        }
                        sb.Append('\n');
                    }
                    for(int i =0; i < m_HiddenCount; i++)
                    {
                        sb.Append($"{m_BiasInputToHidden[i]};");
                    }
                    sb.Append('\n');
                    for(int i =0; i < m_OutputCount; i++)
                    {
                        for(int j = 0; j < m_HiddenCount; j++)
                        {
                            sb.Append($"{m_WeightsHiddenToOutput[i][j]};");
                        }
                        sb.Append('\n');
                    }
                    for (int i = 0; i < m_OutputCount; i++)
                    {
                        sb.Append($"{m_BiasHiddenToOutput[i]};");
                    }
                    sb.Append('\n');
                    for (int i = 0; i < m_OutputCount; i++)
                    {
                        sb.Append($"{m_OutputNames[i]};");
                    }
                    sb.Append('\n');
                    sw.Write(sb.ToString());
                    sw.Close();
                }
            }
            catch(Exception e)
            {
                cultureInfo.NumberFormat.NumberDecimalSeparator = originalNumberDecimalSeparator;
                return $"*** ERROR! ***\n{e}";
            }

            cultureInfo.NumberFormat.NumberDecimalSeparator = originalNumberDecimalSeparator;
            return "OK!";
        }
    }
}