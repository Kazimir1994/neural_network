package kazzimir.bortnik.app.cnn.model;

public class FragmentWall {
    private final double density;
    private final double longitudinalSoundWaveSpeed;
    private final double layerThickness;

    public FragmentWall(double density, double longitudinalSoundWaveSpeed, double layerThickness) {
        this.density = density;
        this.longitudinalSoundWaveSpeed = longitudinalSoundWaveSpeed;
        this.layerThickness = layerThickness;
    }

    public double getDensity() {
        return density;
    }

    public double getLongitudinalSoundWaveSpeed() {
        return longitudinalSoundWaveSpeed;
    }

    public double getLayerThickness() {
        return layerThickness;
    }

    @Override
    public String toString() {
        return "FragmentWall{" +
                "density=" + density +
                ", longitudinalSoundWaveSpeed=" + longitudinalSoundWaveSpeed +
                ", layerThickness=" + layerThickness +
                '}';
    }
}
