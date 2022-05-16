package kazzimir.bortnik.app.cnn.model;

import java.util.List;

public class SystemWall {
    private final double waveFrequency;
    private final List<FragmentWall> fragmentsWall;
    private final double answerDB;

    public SystemWall(double waveFrequency, List<FragmentWall> fragmentsWall, double answerDB) {
        this.waveFrequency = waveFrequency;
        this.fragmentsWall = fragmentsWall;
        this.answerDB = answerDB;
    }

    public double getWaveFrequency() {
        return waveFrequency;
    }

    public List<FragmentWall> getSystemWalls() {
        return fragmentsWall;
    }

    public double getAnswerDB() {
        return answerDB;
    }
}
