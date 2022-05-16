package kazzimir.bortnik.app.cnn;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;

public class VaeMNISTAnomaly {

    public static void main(String[] args) throws IOException {
        int minibatchSize = 128;
        int rngSeed = 12345;
        int nEpochs = 5;

        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed);

        Nd4j.getRandom().setSeed(rngSeed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .updater(new Adam(1e-3))
                .weightInit(WeightInit.XAVIER)
                .l2(1e-4)
                .list()
                .layer(new VariationalAutoencoder.Builder()
                        .activation(Activation.LEAKYRELU)
                        .encoderLayerSizes(256, 256)
                        .decoderLayerSizes(256, 256)
                        .pzxActivationFunction(Activation.IDENTITY)
                        .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                        .nIn(28 * 28)
                        .nOut(32)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(100));
        System.out.println(trainIter.next());
        for (int i = 0; i < 30; i++) {
            net.pretrain(trainIter);
            System.out.println("Finished epoch " + (i + 1) + " of " + nEpochs);
        }
/*        org.deeplearning4j.nn.layers.variational.VariationalAutoencoder vae
                = (org.deeplearning4j.nn.layers.variational.VariationalAutoencoder) net.getLayer(0);
        DataSetIterator testIter = new MnistDataSetIterator(minibatchSize, false, rngSeed);
        DataSet ds = testIter.next();
        INDArray features = ds.getFeatures();
        INDArray row = features.getRow(0,true);
        System.out.println(row);
        LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();
        vae.setInput(row, mgr);
        INDArray pzxMeanBest = vae.preOutput(false, mgr);
        System.out.println(pzxMeanBest);
        INDArray reconstructionBest = vae.generateAtMeanGivenZ(pzxMeanBest);
        System.out.println(reconstructionBest);*/

    }

}