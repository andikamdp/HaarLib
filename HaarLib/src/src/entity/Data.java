/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.entity;

import java.io.File;
import org.opencv.core.Mat;

/**
 *
 * @author Andika Mulyawan
 */
public class Data {

    private File dataFile;
    private Mat dataMat;
    private String dataName;
    private int index;
    private float predictResult;

    public Data(String dataName, float predictResult) {
        this.dataName = dataName;
        this.predictResult = predictResult;
    }

    public Data(File dataFile, Mat dataMat, String dataName, int index) {
        this.dataFile = dataFile;
        this.dataMat = dataMat;
        this.dataName = dataName;
        this.index = index;
    }

    public float getPredictResult() {
        return predictResult;
    }

    public void setPredictResult(float predictResult) {
        this.predictResult = predictResult;
    }

    public Data() {
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }

    public File getDataFile() {
        return dataFile;
    }

    public void setDataFile(File dataFile) {
        this.dataFile = dataFile;
    }

    public Mat getDataMat() {
        return dataMat;
    }

    public void setDataMat(Mat dataMat) {
        this.dataMat = dataMat;
    }

    public String getDataName() {
        return dataName;
    }

    public void setDataName(String dataName) {
        this.dataName = dataName;
    }

}
