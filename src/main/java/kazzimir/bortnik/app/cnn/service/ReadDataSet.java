package kazzimir.bortnik.app.cnn.service;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.List;

public interface ReadDataSet {
    List<Double[][]> read(String path, int muf) throws URISyntaxException, IOException;

    DataSet getDataSet(List<Double[][]> dataset);

    INDArray readData(String s) throws URISyntaxException, IOException;
}
