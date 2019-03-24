/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package src.controller;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.net.URL;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.ResourceBundle;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.AnchorPane;
import javafx.stage.DirectoryChooser;
import org.opencv.core.Mat;
import org.opencv.ml.Ml;
import org.opencv.ml.SVM;
import src.entity.Data;
import src.utils.DataTrainingPrep;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class SVMTrainController implements Initializable {

    @FXML
    private Button btnTrain;
    @FXML
    private TextArea txtAreaStatus;
    @FXML
    private Button btnBrowsImage;
    @FXML
    private Label lblImageLocation;
    @FXML
    private AnchorPane apTrainWindow;
    @FXML
    private Button btnBrowsClassification;
    @FXML
    private Label lblClassificationLocation;
    @FXML
    private Button btnSaveClasification;
    @FXML
    private TextField txtBoxFileName;
    @FXML
    private TextField txtBoxSeed;
    @FXML
    private TextField txtBoxWidthImage;
    @FXML
    private TextField txtBoxLowerTreshold;
//
    private int ratio;
    private MainAppController mainAppController;
    private SVM svm;
    private int seed;
    private List<Double> accuracySeedSampleAvg;
    private List<Double> accuracySeedTrainAvg;
    private List<Double> precisionSeedSampleAvg;
    private List<Double> precisionSeedTrainAvg;
    private List<Double> recallSeedSampleAvg;
    private List<Double> recallSeedTrainAvg;
    private List<Double> falsePositiveSeedSampleAvg;
    private List<Double> falsePositiveSeedTrainAvg;
    private List<Double> f1ScoreSeedSampleAvg;
    private List<Double> f1ScoreSeedTrainAvg;
    private List<SVM> svmList;
    private double treshold;
    private String res;

    private List<Data> trainingResult;
    private List<Data> sampleResult;
//

    /**
     * Initializes the controller class.
     *
     * @param url
     * @param rb
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        ratio = 30;
        accuracySeedTrainAvg = new ArrayList<>();
        accuracySeedSampleAvg = new ArrayList<>();
        //
        precisionSeedTrainAvg = new ArrayList<>();
        precisionSeedSampleAvg = new ArrayList<>();
        //
        recallSeedTrainAvg = new ArrayList<>();
        recallSeedSampleAvg = new ArrayList<>();
        //
        falsePositiveSeedSampleAvg = new ArrayList<>();
        falsePositiveSeedTrainAvg = new ArrayList<>();
        //
        f1ScoreSeedSampleAvg = new ArrayList<>();
        f1ScoreSeedTrainAvg = new ArrayList<>();
        //
        trainingResult = new ArrayList<>();
        sampleResult = new ArrayList<>();
    }

    public void initSvm() {
        svm = SVM.create();
        svm.setKernel(SVM.LINEAR);
        svm.setType(SVM.C_SVC);
//        svm = SVM.load("E:\\TA\\New_Folder\\backgroud_substraction\\lib\\dist\\res 48\\" + txtBoxFileName.getText() + seed + "_48.xml");

    }

    /**
     * ######################################################################
     * OnClick Action memulai Training diikuti Testing
     * training dan testing akan dilakukan sebanyak 10 kali dengan data teracak setiap training
     * training dengan deskripsi gambar edge dan hog
     * var:
     *
     */
    @FXML
    private void trainOnClick(ActionEvent event) {
        res = "";
        //
        treshold = Double.valueOf(txtBoxLowerTreshold.getText());
        //
        accuracySeedTrainAvg.clear();
        accuracySeedSampleAvg.clear();
        //
        precisionSeedTrainAvg.clear();
        precisionSeedSampleAvg.clear();
        //
        recallSeedTrainAvg.clear();
        recallSeedSampleAvg.clear();
        //
        falsePositiveSeedSampleAvg.clear();
        falsePositiveSeedTrainAvg.clear();
        //
        f1ScoreSeedSampleAvg.clear();
        f1ScoreSeedTrainAvg.clear();
        //
        txtAreaStatus.setText("");
        res += "Lokasi Data : " + lblImageLocation.getText() + " \n";
        svmList = new ArrayList<>();
        int seedC = Integer.valueOf(txtBoxSeed.getText());
        for (int i = 0; i < seedC; i++) {
            seed = i;
            initSvm();
            trainingResult.add(new Data("SVM Training Iterasi ", (i + 1)));
            sampleResult.add(new Data("SVM Testing Iterasi ", (i + 1)));
            res += "Waktu Mulai seed " + LocalTime.now() + " \n\n";
            System.out.println(i + " " + LocalTime.now());
            System.out.println(i + " " + LocalTime.now());
            res += "Train SVM Iterasi " + (i + 1) + " \n";
            svmEdgeRandom();

            System.out.println(i + " " + LocalTime.now());
            svmList.add(svm);
            res += "Waktu Selesai seed " + LocalTime.now() + " \n\n";
        }
        rataRataAkurasiSeed();
        System.out.println("");
        printPredictedResult();
        txtAreaStatus.setText(res);
    }

    /**
     * ######################################################################
     * method set Main Controller
     *
     */
    void setMainController(MainAppController aThis
    ) {
        mainAppController = aThis;
    }

    /**
     * ######################################################################
     * method untuk train SVM dengan deskripsi gambar Edge
     * var:
     * SVM svm : variabel klasifikasi dengan SVM
     * Mat trainingDataMat : variabel penampung data training
     * Mat labelsMat : variabel penampung label data training
     * Mat sampleDataMat : variabel penampung data testing
     * int rows :
     * File file : variabel penampung lokasi file data gambar
     * File[] listOfFiles : variabe penampung isi file data gambar
     * int[][] confusionMatrix : variabel penampung confussion matriks
     */
    public void svmEdgeRandom() {

        File file = new File(lblImageLocation.getText());
        File[] listFiles = file.listFiles();
//######################################################################
        List<Integer> index = getRandomIndex(listFiles[0].listFiles().length);
        List<String> labels = new ArrayList<>();
        //
        File files;
        int rows = 0;
        Mat dataTraining = new Mat();
        List<Data> trainingDataMat = new ArrayList<>();
        List<Data> sampleDataMat = new ArrayList<>();
        Mat labelsMat = new Mat();
        double width = Double.valueOf(txtBoxWidthImage.getText());
        for (int i = 0; i < listFiles.length; i++) {
            files = listFiles[i];
            trainingDataMat.addAll(DataTrainingPrep.getDataSVMEdge(files.getAbsolutePath(), index, true, width, treshold));
            sampleDataMat.addAll(DataTrainingPrep.getDataSVMEdge(files.getAbsolutePath(), index, false, width, treshold));
            if (i == 0) {
                rows = trainingDataMat.size();
            }
            labelsMat.push_back(DataTrainingPrep.getLabel(rows, i));
            labels.add(files.getName());
        }
        dataTraining = DataTrainingPrep.getDataMat(trainingDataMat);

        res += "Waktu Mulai Training " + LocalTime.now() + " \n";
        System.out.println("Waktu Mulai Training " + LocalTime.now());
        svm.train(dataTraining, Ml.ROW_SAMPLE, labelsMat);
        res += "Waktu Selesai Training " + LocalTime.now() + " \n\n";
        System.out.println("Waktu Selesai Training " + LocalTime.now());
        //######################################################################
        int[][] confusionMatrix = predictClassifier(labels, trainingDataMat, true);
        confusionMatriks(confusionMatrix, trainingDataMat.size(), true);
        //######################################################################
        confusionMatrix = predictClassifier(labels, sampleDataMat, false);
        confusionMatriks(confusionMatrix, sampleDataMat.size(), false);
        System.gc();
    }

    /**
     * ######################################################################
     * OnClick Action untuk memperoleh lokasi data gambar
     * var:
     * DirectoryChooser brows :
     * File Path :
     */
    @FXML
    private void browsImageOnClick(ActionEvent event) {
        DirectoryChooser brows = new DirectoryChooser();
        brows.setTitle("Buka Folder Data Training");
        File Path = brows.showDialog(apTrainWindow.getScene().getWindow());
        if (Path != null) {
            lblImageLocation.setText(Path.getAbsolutePath());
        }

    }

    /**
     * ######################################################################
     * method untuk menghitung akurasi training
     * var:
     * float TP, TN, FP, FN : fariabel penampung nilai True, Positif, False, Negatif
     * float precision, recall, accuracy :
     *
     * @param predict
     * @param jumlahData
     * @param train
     */
    public void confusionMatriks(int[][] predict, int jumlahData, boolean train) {
        float[] TP = new float[predict.length],
                TN = new float[predict.length],
                FP = new float[predict.length],
                FN = new float[predict.length];
        float[] precision = new float[predict.length],
                recall = new float[predict.length],
                accuracy = new float[predict.length],
                falsePositiveRate = new float[predict.length],
                f1Score = new float[predict.length];
        float avgAccuracy = 0,
                avgRecall = 0,
                avgPrecision = 0,
                avgfalsePositiveRate = 0,
                avgf1Score = 0;
        for (int i = 0; i < predict.length; i++) {
            for (int j = 0; j < predict.length; j++) {
                if (i == j) {
                    TP[i] = predict[i][j];
                } else if (i != j) {
                    FP[i] += predict[i][j];
                    FN[i] += predict[j][i];
                }
            }
            TN[i] = jumlahData - (TP[i] + FP[i] + FN[i]);
        }
        for (int i = 0; i < predict.length; i++) {
            precision[i] = TP[i] / (TP[i] + FP[i]);
            recall[i] = TP[i] / (TP[i] + FN[i]);
            accuracy[i] = (TP[i] + TN[i]) / jumlahData;
            falsePositiveRate[i] = FP[i] / (TN[i] + FP[i]);
            f1Score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]);
            avgAccuracy += accuracy[i];
            avgRecall += recall[i];
            avgPrecision += precision[i];
            avgfalsePositiveRate += falsePositiveRate[i];
            avgf1Score += f1Score[i];
        }
        avgAccuracy /= predict.length;
        avgRecall /= predict.length;
        avgPrecision /= predict.length;
        avgfalsePositiveRate /= predict.length;
        avgf1Score /= predict.length;
        if (train) {
            res += "Evaluasi dengan data training \n\n";
        } else {
            res += "Evaluasi dengan data testing \n\n";
        }
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 6; k++) {
                res += predict[i][k] + " ";
            }
            res += " \n";
        }
        res += " \n";
        res += "TP: " + Arrays.toString(TP) + " \n";
        res += "TN: " + Arrays.toString(TN) + " \n";
        res += "FP: " + Arrays.toString(FP) + " \n";
        res += "FN: " + Arrays.toString(FN) + " \n";
        res += "precision: " + Arrays.toString(precision) + " \n";
        res += "recall: " + Arrays.toString(recall) + " \n";
        res += "accuracy: " + Arrays.toString(accuracy) + " \n";
        res += "falsePositiveRate: " + Arrays.toString(falsePositiveRate) + " \n";
        res += "F1Score: " + Arrays.toString(f1Score) + " \n\n";
        res += "Avg accuracy: " + avgAccuracy + " \n\n";
        res += "Avg recall: " + avgRecall + " \n\n";
        res += "Avg precision: " + avgPrecision + " \n\n";
        res += "avg falsePositiveRate: " + avgfalsePositiveRate + " \n\n";
        res += "avg F1Score: " + avgf1Score + " \n\n\n";
//        res += "overAllAccuracy: " + overAllAccuracy + " \n\n\n\n";

        if (train) {
            accuracySeedTrainAvg.add(Double.valueOf(avgAccuracy));
            precisionSeedTrainAvg.add(Double.valueOf(avgPrecision));
            recallSeedTrainAvg.add(Double.valueOf(avgRecall));
            falsePositiveSeedTrainAvg.add(Double.valueOf(avgfalsePositiveRate));
            f1ScoreSeedTrainAvg.add(Double.valueOf(avgf1Score));
        } else {
            accuracySeedSampleAvg.add(Double.valueOf(avgAccuracy));
            precisionSeedSampleAvg.add(Double.valueOf(avgPrecision));
            recallSeedSampleAvg.add(Double.valueOf(avgRecall));
            falsePositiveSeedSampleAvg.add(Double.valueOf(avgfalsePositiveRate));
            f1ScoreSeedSampleAvg.add(Double.valueOf(avgf1Score));
        }
    }

    /**
     * ######################################################################
     * method untuk menghitung rata-rata akurasi
     * var:
     * float i : menghitung jumlah akurasi
     */
    private void rataRataAkurasiSeed() {
        float accTrn = 0, accSmpl = 0, fPTrn = 0, fPSmp = 0, f1ScoreSmpl = 0, f1ScoreTrn = 0, prcTrn = 0, prcSmpl = 0, rclTrn = 0, rclSmpl = 0;
        for (int j = 0; j < accuracySeedTrainAvg.size(); j++) {
            accTrn += accuracySeedTrainAvg.get(j);
            accSmpl += accuracySeedSampleAvg.get(j);
            //
            fPTrn += falsePositiveSeedTrainAvg.get(j);
            fPSmp += falsePositiveSeedSampleAvg.get(j);
            //
            prcTrn += precisionSeedTrainAvg.get(j);
            prcSmpl += precisionSeedSampleAvg.get(j);
            //
            rclTrn += recallSeedTrainAvg.get(j);
            rclSmpl += recallSeedSampleAvg.get(j);
            //
            f1ScoreTrn += f1ScoreSeedTrainAvg.get(j);
            f1ScoreSmpl += f1ScoreSeedSampleAvg.get(j);
        }
        res += "Average accuracy Train: " + (accTrn / accuracySeedTrainAvg.size()) + " \n";
        res += "Average accuracy Sample: " + (accSmpl / accuracySeedSampleAvg.size()) + " \n\n";
        //
        res += "Average precision Train: " + (prcTrn / precisionSeedTrainAvg.size()) + " \n";
        res += "Average precision Sample: " + (prcSmpl / precisionSeedSampleAvg.size()) + " \n\n";
        //
        res += "Average recall Train: " + (rclTrn / recallSeedTrainAvg.size()) + " \n";
        res += "Average recall Sample: " + (rclSmpl / recallSeedSampleAvg.size()) + " \n\n";
        //
        res += "Average FP Rate Train: " + (fPTrn / falsePositiveSeedTrainAvg.size()) + " \n";
        res += "Average FP Rate Sample: " + (fPSmp / falsePositiveSeedSampleAvg.size()) + " \n\n";
        //
        res += "Average F1 Score Train: " + (f1ScoreTrn / f1ScoreSeedTrainAvg.size()) + " \n";
        res += "Average F1 Score  Sample: " + (f1ScoreSmpl / f1ScoreSeedSampleAvg.size()) + " \n\n";
    }

    /**
     * ######################################################################
     * method untuk train SVM dengan data gambar kombinasi jari terangkat
     * param:
     * int jumlahData : variabel penampung jumlah data training perClass(label)
     * var:
     * List<Integer> index : variabel penampung urutan angka sebanyak jumlahData
     * List<Integer> indexSample : variabel penampung angka random
     * Random rand : variabel penampung jumlah data training
     * int numberOfElements :
     *
     *
     * @param jumlahData
     * @return
     */
    public List<Integer> getRandomIndex(int jumlahData) {
        List<Integer> index = new ArrayList<>();
        List<Integer> indexSample = new ArrayList<>();
        for (int i = 0; i < jumlahData; i++) {
            index.add(i);
        }
        Random rand = new Random(seed);
        int numberOfElements = (int) (((double) ratio / 100.0) * (double) jumlahData);
        for (int i = 0; i < numberOfElements; i++) {
            int randomIndex = rand.nextInt(index.size());
            indexSample.add(index.get(randomIndex));
            index.remove(randomIndex);
        }
        Collections.sort(indexSample);
        return indexSample;
    }

    /**
     * ######################################################################
     * method untuk
     * param:
     *
     * var:
     *
     * @param fileName
     * @param labels
     * @param data
     * @return
     */
    public int[][] predictClassifier(List<String> labels, List<Data> data, boolean train) {
        int[][] confusionMatrix = new int[6][6];
        for (int j = 0; j < data.size(); j++) {
            float label = svm.predict(data.get(j).getDataMat());
            data.get(j).setPredictResult(label);
//            res += data.get(j).getDataName() + "_" + label + " \n";
            int l = 0, m = 0;
            if (data.get(j).getDataName().contains(labels.get(0).substring(0, 1))) {
                l = 0;
            } else if (data.get(j).getDataName().contains(labels.get(1).substring(0, 1))) {
                l = 1;
            } else if (data.get(j).getDataName().contains(labels.get(2).substring(0, 1))) {
                l = 2;
            } else if (data.get(j).getDataName().contains(labels.get(3).substring(0, 1))) {
                l = 3;
            } else if (data.get(j).getDataName().contains(labels.get(4).substring(0, 1))) {
                l = 4;
            } else if (data.get(j).getDataName().contains(labels.get(5).substring(0, 1))) {
                l = 5;
            }
            if (label == 0) {
                m = 0;
            } else if (label == 1) {
                m = 1;
            } else if (label == 2) {
                m = 2;
            } else if (label == 3) {
                m = 3;
            } else if (label == 4) {
                m = 4;
            } else if (label == 5) {
                m = 5;
            }
            if (train) {
                trainingResult.add(new Data(data.get(j).getDataName(), label));
            } else {
                sampleResult.add(new Data(data.get(j).getDataName(), label));
            }
            confusionMatrix[m][l] += 1;
        }
        res += " \n\n";
        return confusionMatrix;
    }

    @FXML
    private void browsClassificationOnClick(ActionEvent event) {
        DirectoryChooser brows = new DirectoryChooser();
        brows.setTitle("Buka Folder Classification Save Lokasi");
        File Path = brows.showDialog(apTrainWindow.getScene().getWindow());
        if (Path != null) {
            lblClassificationLocation.setText(Path.getAbsolutePath());
        }
    }

    @FXML
    private void saveClasssificationOnClilck(ActionEvent event) {
        try {
            String string = txtAreaStatus.getText();
            File file = new File(lblClassificationLocation.getText() + "\\" + txtBoxFileName.getText() + "_" + txtBoxLowerTreshold.getText() + "_" + txtBoxWidthImage.getText() + ".txt");
            try (
                    BufferedReader reader = new BufferedReader(new StringReader(string));
                    PrintWriter writer = new PrintWriter(new FileWriter(file));) {
                reader.lines().forEach(new Consumer<String>() {
                    @Override
                    public void accept(String line) {
                        writer.println(line);
                    }
                });
            }
        } catch (IOException ex) {
            Logger.getLogger(SVMTrainController.class.getName()).log(Level.SEVERE, null, ex);
        }
        for (int i = 0; i < svmList.size(); i++) {
            svmList.get(i).save(lblClassificationLocation.getText() + "\\" + txtBoxFileName.getText() + "_" + i + "_" + txtBoxLowerTreshold.getText() + "_" + txtBoxWidthImage.getText() + ".xml");
        }
    }

    private void printPredictedResult() {
        res += " \n\n\n\n";
        for (Data data : trainingResult) {
            res += data.getDataName() + "_" + data.getPredictResult() + " \n";
        }
        res += " \n\n\n\n";
        for (Data data : sampleResult) {
            res += data.getDataName() + "_" + data.getPredictResult() + " \n";
        }
    }
}
