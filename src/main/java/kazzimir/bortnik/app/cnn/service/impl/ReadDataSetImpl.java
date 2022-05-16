package kazzimir.bortnik.app.cnn.service.impl;

import kazzimir.bortnik.app.cnn.model.FragmentWall;
import kazzimir.bortnik.app.cnn.model.SystemWall;
import kazzimir.bortnik.app.cnn.service.ReadDataSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ReadDataSetImpl implements ReadDataSet {
    private static ReadDataSetImpl dataSet = new ReadDataSetImpl();

    public static ReadDataSetImpl getInstants() {
        return dataSet;
    }

    private ReadDataSetImpl() {

    }

    @Override
    public List<Double[][]> read(String path, int muf) throws URISyntaxException, IOException {
        Path pathDataSet = getPathDataSet(path);
        return Files.lines(pathDataSet)
                .map(line -> line.split(";"))
                .map(strings -> buildSystemWall(strings, muf))
                .collect(Collectors.toList());
    }

    @Override
    public DataSet getDataSet(List<Double[][]> dataset) {
        INDArray datasetInput = Nd4j.create(dataset.size(), dataset.get(0)[0].length);
        INDArray datasetOutput = Nd4j.create(dataset.size(), dataset.get(0)[1].length);
        Double max = Collections.max(dataset, Comparator.comparing(doubles -> doubles[1][0]))[1][0];
        IntStream.range(0, dataset.size())
                .forEach(index -> {
                    Double[][] doubles = dataset.get(index);
                    datasetInput.putRow(index, Nd4j.createFromArray(doubles[0]));
                    datasetOutput.putRow(index, Nd4j.createFromArray((doubles[1][0] / max)));
                });
        DataSet dataSet = new DataSet();
        dataSet.setFeatures(datasetInput);
        dataSet.setLabels(datasetOutput);
        return dataSet;
    }

    @Override
    public INDArray readData(String path) throws URISyntaxException, IOException {
        List<Double[]> collect = Files.lines(getPathDataSet(path))
                .map(s -> s.split(","))
                .map(strings -> {
                    Double[] inputData = new Double[strings.length];
                    for (int i = 0; i < strings.length; i++) {
                        inputData[i] = Double.parseDouble(strings[i]);
                    }
                    return inputData;
                }).collect(Collectors.toList());
        INDArray indArray = Nd4j.create(collect.size(), collect.get(0).length);
        IntStream.range(0, collect.size())
                .forEach(value -> {
                    indArray.putRow(value, Nd4j.createFromArray(collect.get(value)));
                });
        return indArray;
    }

    private Path getPathDataSet(String path) throws URISyntaxException {
        return Paths.get(getClass().getClassLoader()
                .getResource(path).toURI());
    }

    private Double[][] buildSystemWall(String[] data, int muf) {
        Double[] inputData = new Double[data.length - muf];
        for (int i = 0; i < data.length - muf; i++) {
            inputData[i] = Double.parseDouble(data[i + muf - 1]);
        }
        Double[] outputData = {Double.parseDouble(data[data.length - 1])};
        Double[] pref = {Double.parseDouble(data[0])};
        return new Double[][]{inputData, outputData, pref};
    }
   /* private SystemWall buildSystemWall(String[] data) {
        double waveFrequency = Double.parseDouble(data[0]);
        double answerDB = Double.parseDouble(data[data.length - 1]);
        List<FragmentWall> fragmentsWall = getFragmentsWall(data);
        return new SystemWall(waveFrequency, fragmentsWall, answerDB);
    }

    private List<FragmentWall> getFragmentsWall(String[] data) {
        List<FragmentWall> fragmentWalls = new ArrayList<>();
        for (int i = 0; i < data.length / 3; i++) {
            double density = Double.parseDouble(data[(i * 3) + 1]);
            double longitudinalSoundWaveSpeed = Double.parseDouble(data[(i * 3) + 2]);
            double layerThickness = Double.parseDouble(data[(i * 3) + 3]);
            FragmentWall fragmentWall = new FragmentWall(density, longitudinalSoundWaveSpeed, layerThickness);
            fragmentWalls.add(fragmentWall);
        }
        return fragmentWalls;
    }*/
}
