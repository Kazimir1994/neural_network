package kazzimir.bortnik.app.cnn;

import kazzimir.bortnik.app.cnn.service.impl.ReadDataSetImpl;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.SamplingDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Application {
    public static void main(String[] args) throws URISyntaxException, IOException {

        extracted1();
    }

    private static void extracted1() throws IOException, URISyntaxException {
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("./model4500.h5"), true);
        ReadDataSetImpl instants = ReadDataSetImpl.getInstants();
        List<Double[][]> dataset = instants.read("./data/dataset.txt", 1);
        DataSet dataSet = instants.getDataSet(dataset);
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);

        INDArray indArray = instants.readData("./data/predict.txt");

        DataSet dataSet2 = new DataSet();
        dataSet2.setFeatures(indArray);
        dataSet2.setLabels(indArray);
        normalizer.transform(dataSet2);

        INDArray indArray2 = instants.readData("./data/predict.txt");
        INDArray output = model.output(indArray);
        output.muli(99.6);
        for (int i = 0; i < 31; i++) {
            System.out.println(indArray2.getRow(i));
            System.out.println(output.getRow(i));
            System.out.println("_______________________________________________");
        }
    }

    private static void extracted() throws URISyntaxException, IOException {
     /*   MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(3)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.LEAKYRELU)
                .updater(new Adam(0.001))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(new DenseLayer.Builder().nIn(31).nOut(250).build())
                .layer(new DenseLayer.Builder().nIn(250).nOut(700).build())
                .layer(new DenseLayer.Builder().nIn(700).nOut(700).build())
                .layer(new DenseLayer.Builder().nIn(700).nOut(255).build())
                .layer(new DenseLayer.Builder().nIn(255).nOut(900).build())
                .layer(new DenseLayer.Builder().nIn(900).nOut(31).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(31)
                        .activation(Activation.LEAKYRELU)
                        .nOut(1).build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));*/

        ReadDataSetImpl instants = ReadDataSetImpl.getInstants();
        List<Double[][]> dataset = instants.read("./data/dataset.txt", 1);
        DataSet dataSet = instants.getDataSet(dataset);
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);
        dataSet.shuffle();
        MultiLayerNetwork model = MultiLayerNetwork.load(new File("./model.h5"), true);
        model.setListeners(new ScoreIterationListener(100));
        for (int i = 0; i < 6000; i++) {
            model.fit(dataSet);
            if ((i % 500) == 0) {
                model.save(new File("./model" + i + ".h5"));
            }
        }
   //     model.save(new File("./model.h5"));
        for (int i = 0; i < dataset.size(); i++) {
            INDArray indArray = Nd4j.create(1, 31);
            indArray.putRow(0, dataSet.getFeatures().getRow(i));
            INDArray output = model.output(indArray);
            INDArray row = output.getRow(0);
            System.out.println((row.toDoubleVector()[0]) + "  " + dataSet.getLabels().getRow(i).toDoubleVector()[0]);
        }

        //evaluate the model on the test set
        Evaluation eval = new Evaluation();
        INDArray output = model.output(dataSet.getFeatures());
        eval.eval(dataSet.getLabels(), output);
        System.out.println(eval.stats());
    }

    private static void data() throws URISyntaxException, IOException {
        ReadDataSetImpl instants = ReadDataSetImpl.getInstants();
        List<Double[][]> dataset = instants.read("./data/datnew", 2);
        DataSet dataSet = instants.getDataSet(dataset);
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);
        dataSet.setLabels(dataSet.getFeatures());
        dataSet.shuffle();
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(22)
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.UNIFORM)
                .l2(1e-4)
                .list()
                .layer(new VariationalAutoencoder.Builder()
                        .activation(Activation.LEAKYRELU)
                        .encoderLayerSizes(850, 950, 1000)
                        .decoderLayerSizes(850, 950, 1000)
                        .pzxActivationFunction(Activation.IDENTITY)
                        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(30)
                        .nOut(5)
                        .build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(100));

        DataSetIterator dataSetIterator = new SamplingDataSetIterator(dataSet, 1207, 1207);
        net.pretrain(dataSetIterator, 5000);

        INDArray row = dataSet.getFeatures().getRow(0, true);
        System.out.println(net.output(row));
        System.out.println(net.output(row));

        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);
        LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
        vae.setInput(row, mgr);
        INDArray output = vae.preOutput(false, mgr);
        System.out.println(output);
        vae.setInput(row, mgr);
        INDArray output2 = vae.preOutput(false, mgr);
        System.out.println(output2);
        // net.save(new File("./ENCODER.h5"));
    }

    private final static void preprocessingData() throws IOException, URISyntaxException {
        ReadDataSetImpl instants = ReadDataSetImpl.getInstants();
        List<Double[][]> dataset = instants.read("./data/dataset.txt", 2);
        DataSet dataSet = instants.getDataSet(dataset);
        DataNormalization normalizer = new NormalizerMinMaxScaler();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);
        MultiLayerNetwork net = MultiLayerNetwork.load(new File("./ENCODER.h5"), true);
        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);
        LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
        vae.setInput(dataSet.getFeatures(), mgr);
        INDArray output = vae.preOutput(false, mgr);
        FileWriter myWriter = new FileWriter("filename.txt");
        for (int i = 0; i < 1271; i++) {
            double[] doubles = output.getRow(i, true).toDoubleVector();
            String collect = Arrays.stream(doubles)
                    .mapToObj(String::valueOf)
                    .collect(Collectors.joining(";"));
            myWriter.write(dataset.get(i)[2][0] + ";" + collect + ";" + dataset.get(i)[1][0]);
            myWriter.write("\r\n");
        }
    }
}
 /*   INDArray indArray = Nd4j.create(1, 31);
        indArray.putRow(0, dataSet.getFeatures().getRow(0));
                INDArray row1 = dataSet.getLabels().getRow(0);
                System.out.println(row1);
                INDArray output = model.output(indArray);
                System.out.println(output);
                INDArray row = output.getRow(0);
                System.out.println(row.toDoubleVector()[0] * 72.6);
                Evaluation eval = new Evaluation();
                INDArray output2 = model.output(dataSet.getFeatures());
                eval.eval(dataSet.getLabels(), output2);
                System.out.println(eval.stats());*/

  /*      DataSet dataSet = new DataSet();
        dataSet.setFeatures(datasetForProcessing);
        INDArray fromArray = Nd4j.createFromArray(700.0);

        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(dataSet);
        normalizer.transform(dataSet);
        normalizer.transform(fromArray);
        INDArray row = dataSet.getFeatures().getRow(0);
        System.out.println(row);
        System.out.println(fromArray);*/
