using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Encog;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.Neural.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.NeuralData;


namespace ConsoleApp1
{
    class Program
    {
        private static readonly double AcceptableError = 0.0001;
        private static readonly int MaxEpoch = 5000;
        private static readonly double[][] AndInput ={
            new double[2] { 0.0, 0.0 },
            new double[2] { 1.0, 0.0 },
            new double[2] { 0.0, 1.0 },
            new double[2] { 1.0, 1.0 } };
        private static readonly double[][] AndIdeal = {
            new double[1] { 1.0 },
            new double[1] { 0.0 },
            new double[1] { 0.0 },
            new double[1] { 1.0 } };

        static void Main(string[] args)
        {
            INeuralDataSet trainingSet = new BasicNeuralDataSet(AndInput, AndIdeal);
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 2));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 1));
            network.Structure.FinalizeStructure();
            network.Reset();

            ITrain train = new ResilientPropagation(network, trainingSet);
            
            int epoch = 1;
            do
            {
                train.Iteration();
                Console.WriteLine($"Epoch no {epoch}. Error: {train.Error}");
                epoch++;
            } while ((epoch < MaxEpoch) && (train.Error > AcceptableError));

            Console.WriteLine("\nAnd function Results:");
            foreach (IMLDataPair pair in trainingSet)
            {
                IMLData output = network.Compute(pair.Input);
                Console.WriteLine($"{pair.Input[0]} AND {pair.Input[1]} should be: {pair.Ideal[0]} actual value is: {output[0]}");
            }


            Console.ReadKey();
        }

    }
}
