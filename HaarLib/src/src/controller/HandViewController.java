/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the templateasdasdasd in the editor.
 */
package src.controller;

import java.io.File;
import src.Utils;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ResourceBundle;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.videoio.VideoCapture;
import src.utils.Preprocessing;

/**
 * FXML Controller class
 *
 * @author Andika Mulyawan
 */
public class HandViewController implements Initializable {

    @FXML
    private ImageView layarBW;
    @FXML
    private ImageView layarEdge;
    @FXML
    private ImageView layarMain;
    @FXML
    private Button btnStartCamera;
    @FXML
    private Button btnUpdateCamera;
    @FXML
    private TextField txtFileLocation;
    @FXML
    private TextField txtFileName;
    @FXML
    private TextField txtS;
    @FXML
    private TextField txtValue;
    @FXML
    private TextField txtMainFramePoint;
    @FXML
    private TextField txtPredictedResult;
    @FXML
    private ComboBox<String> cmbClassifier;
    @FXML
    private ComboBox<String> cmbTresholdType;
//
    private MainAppController mainAppController;
    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    private int i;
    private SVM svmJariTerangkatHOG, svmBisindoEdge;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        capture = new VideoCapture();
        cameraActive = false;
        i = 0;
        svmBisindoEdge = SVM.load("E:\\TA\\Bisindo.xml");
        svmJariTerangkatHOG = SVM.load("E:\\TA\\hCoba.xml");
        ObservableList<String> typeTreshold = FXCollections.observableArrayList();
        typeTreshold.add("Bynary");
        typeTreshold.add("Bynary Inverse");
        ObservableList<String> typeClassifier = FXCollections.observableArrayList();
        typeClassifier.add("Bisindo Edge");
        typeClassifier.add("Jari Hog");
        cmbTresholdType.setItems(typeTreshold);
        cmbClassifier.setItems(typeClassifier);
    }

    public void setMainController(MainAppController aThis) {
        mainAppController = aThis;
    }

//######################################################################
    /**
     * Method awal untuk membuka kamera dan memanggil method
     * var:
     * boolean cameraActive : titik saat ini
     * VideoCapture capture : titik sebelumnya
     * Runnable frameGrabber :
     * Mat frame :
     * ScheduledExecutorService timer :
     * Button btnStartCamera :
     */
    @FXML
    private void startCameraOnClick(ActionEvent event) {
        if (!cameraActive) {
            capture.open(0);
            if (this.capture.isOpened()) {
                cameraActive = true;
                Runnable frameGrabber = new Runnable() {
                    @Override
                    public void run() {
                        Mat frame = null;
                        try {
                            frame = grabFrame();
                            start(frame);
                        } catch (Exception e) {
                            System.out.println("startCameraOnClick " + e);
                        }
                    }
                };
                this.timer = Executors.newSingleThreadScheduledExecutor();
                timer.scheduleAtFixedRate(frameGrabber, 0, 33,
                        TimeUnit.MILLISECONDS);
                btnStartCamera.setText("stop Camera");
            } else {
                System.err.println("tidak dapat membuka kamera");
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.btnStartCamera.setText("Start Camera");
            // stop the timer
            this.stopAcquisition();
        }
    }

//######################################################################
    /**
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
     * Mat frame : titik saat ini
     * Image imageToMat : titik sebelumnya
     * Mat hand :
     * Mat tresholded :
     * double x, x_, y, y_:
     * Point p, p_ :
     * List<MatOfPoint> countour :
     * List<MatOfInt4> devOfInt4s :
     * List<MatOfPoint> devOfPoints :
     */
    private void start(Mat frame) {
        Core.flip(frame, frame, 1);
        Preprocessing.drawRect(frame);
        updateImageView(layarMain, frame);
        Mat hand = Preprocessing.getBox(frame.clone());
        //
        Mat tresholded;
        if (cmbTresholdType.getValue().equals("Bynary")) {
            tresholded = Preprocessing.segment(hand.clone(), Double.valueOf(txtValue.getText()));
        } else {
            tresholded = Preprocessing.segmentInvers(hand.clone(), Double.valueOf(txtValue.getText()));
        }
        Point[] extremePoint = getExtremePoint(tresholded);
        Mat handView = Preprocessing.getEdge_2(hand.clone());
        updateImageView(layarEdge, handView);
        //
        Mat handPredict = Preprocessing.getBox(hand.clone(), extremePoint[0], extremePoint[1]);

//
        hand = Preprocessing.drawRect(hand, extremePoint[0], extremePoint[1]);
        updateImageView(layarBW, hand);
    }
//######################################################################

    /**
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
     *
     */
    public Point[] getExtremePoint(Mat tresholded) {
        List<MatOfPoint> contour = Preprocessing.getContour(tresholded);
        List<MatOfInt4> devOfInt4s = Preprocessing.getDevectIndexPoint(contour);
        List<Point> pointContourSorted = Preprocessing.toListContour(contour.get(0));
        Point[] extremePoint = new Point[2];
        //ambil titik ekstreme
        double x, x_, y, y_;
        pointContourSorted = Preprocessing.sortPointByX(pointContourSorted);
        x = pointContourSorted.get(0).x;
        x_ = pointContourSorted.get(pointContourSorted.size() - 1).x;
        pointContourSorted = Preprocessing.sortPointByY(pointContourSorted);
        y = pointContourSorted.get(0).y;
        y_ = pointContourSorted.get(pointContourSorted.size() - 1).y;
        Point p = new Point(x - 10, y - 10);
        Point p_ = new Point(x_ + 10, y_);
        extremePoint[0] = p;
        extremePoint[1] = p_;
        return extremePoint;
    }
    //######################################################################

    /**
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
     *
     */
    public void getPredictedResult(Mat hand) {
        if (cmbClassifier.getValue().equals("Bisindo Edge")) {
            hand = getDataSVMEdgeDELETE(hand);
            txtPredictedResult.setText(String.valueOf(svmBisindoEdge.predict(hand)));
        } else if (cmbClassifier.getValue().equals("Jari Hog")) {
            hand = getDataSVMHogDELETE(hand);
            txtPredictedResult.setText(String.valueOf(svmJariTerangkatHOG.predict(hand)));
        }
    }

    //######################################################################
    /**
     * method untuk memeriksa memperoleh data training berdasarkan fitur HOG
     * var:
     * File folder : lokasi direktori data gambar
     * File[] listOfFiles :
     * Mat trainingDataMat :
     * Mat hand :
     * float[] trainingData:
     */
    public Mat getDataSVMHogDELETE(Mat predict) {
        Mat trainingDataMat;
        trainingDataMat = new Mat(1, 192780, CvType.CV_32FC1);
        HOGDescriptor gDescriptor = new HOGDescriptor();
        Imgproc.resize(predict, predict, new Size(192, 144));
        MatOfFloat descriptors = new MatOfFloat();
        gDescriptor.compute(predict, descriptors);
        float[] trainingData = descriptors.toArray();
        for (int j = 0; j < trainingData.length; j++) {
            trainingData[j] = Math.round(trainingData[j] * 100000) / 100;
        }
        trainingDataMat.put(0, 0, trainingData);

        return trainingDataMat;
    }
//######################################################################

    /**
     * method untuk memeriksa memperoleh data training berdasarkan fitur garis tepi
     * var:
     * File folder : lokasi direktori data gambar
     * File[] listOfFiles :
     * Mat trainingDataMat :
     * Mat hand :
     * float[] trainingData:
     */
    public Mat getDataSVMEdgeDELETE(Mat predict) {

        Mat trainingDataMat;
        trainingDataMat = new Mat(1, 48 * 64, CvType.CV_32FC1);

        predict = Preprocessing.getEdge(predict);

        float[] trainingData = new float[predict.cols()];
        for (int j = 0; j < predict.cols(); j++) {
            trainingData[j] = (float) predict.get(0, j)[0];
        }
        trainingDataMat.put(0, 0, trainingData);

        return trainingDataMat;
    }
//######################################################################

    /**
     * method button btnUpdateCamera OnClick
     * var:
     * int i : reset nomor urut gambar yang di simpan
     */
    @FXML
    private void updateCameraOnClick(ActionEvent event
    ) {
        i = 0;
    }
    //######################################################################

    /**
     * method untuk menyimpan gambar dalam frame utama
     * nama yang digunakan meruapak nomor urut dari index i
     * var:
     * int 1 : nomor urut gambar yang akan digunakan sebagai nama
     */
    public void captureImage() {
        Imgcodecs.imwrite("E:\\TA\\HandLearnSVM\\handFullTry1\\" + i
                + ".jpg",
                grabFrame());
        i++;
    }

//######################################################################
    /**
     * method untuk menyimpan gambar dalam frame utama
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
     * Point titikA : titik saat ini
     * Point titikB : titik sebelumnya
     *
     */
    public void captureImage(Mat frame) {
        Imgcodecs.imwrite("E:\\TA\\HandLearnSVM\\BISINDO\\" + txtFileLocation.getText() + "\\" + txtFileName.getText() + "_" + i
                + ".jpg",
                frame);
        i++;
    }

//######################################################################
    /**
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
     * VideoCapture capture : titik saat ini
     * ScheduledExecutorService timer : titik sebelumnya
     *
     */
    private void stopAcquisition() {
        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                // log any exception
                System.err.println(
                        "Exception in stopping the frame capture, trying to release the camera now... "
                        + e);
            }
        }
        if (this.capture.isOpened()) {
            // release the camera
            this.capture.release();
        }
    }

//######################################################################
    /**
     * update tampilan pada frame utama
     * var:
     *
     */
    private void updateImageView(ImageView view, Mat image) {
        Image imageToMat;
        imageToMat = Utils.mat2Image(image);
        Utils.onFXThread(view.imageProperty(), imageToMat);
    }

//######################################################################
    /**
     * On application close, stop the acquisition from the camera
     * var:
     *
     */
    protected void setClosed() {
        this.stopAcquisition();
    }

//######################################################################
    /**
     *
     * var:
     * VideoCapture capture :
     *
     */
    private Mat grabFrame() {
        // init everything
        Mat frame = new Mat();
        // check if the capture is open
        if (this.capture.isOpened()) {
            try {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {
                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BayerBG2BGR);

                }

            } catch (Exception e) {
                // log the error
                System.err.println("Exception during the image elaboration: "
                        + e);
            }
        }
        return frame;
    }

//######################################################################
    /**
     * method layarBW OnClick
     * method untuk menukar image pada layarBW ke Main
     * var:
     * Image Mn : titik saat ini
     * ImageView layarBW : titik sebelumnya
     * ImageView layarMain
     */
    @FXML
    private void bwToMn(MouseEvent event) {
        Image Mn = layarBW.getImage();
        layarBW.setImage(layarMain.getImage());
        layarMain.setImage(Mn);
    }

//######################################################################
    /**
     * method layarEdge OnClick
     * method untuk menukar image pada layarEdge ke Main
     * var:
     * Image Mn : titik saat ini
     * ImageView layarEdge : titik sebelumnya
     * ImageView layarMain
     */
    @FXML
    private void edgeToMn(MouseEvent event) {
        Image Mn = layarEdge.getImage();
        layarEdge.setImage(layarMain.getImage());
        layarMain.setImage(Mn);
    }

    /*
    method untuk mencoba pada gambar
     */
//######################################################################
    /**
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * method menggambil gambar(image capture)melalui button
     * var:
     * Point titikA : titik saat ini
     * Point titikB : titik sebelumnya
     *
     */
    @FXML
    private void capturePictureOnSction(ActionEvent event) {

    }

//######################################################################
    /**
     * method untuk memeriksa apakah titik saat ini lebih tinggi dari titik sebelumnya
     * semakin kecil nilai titik pada frame semakin tinggi
     * var:
     * Point titikA : titik saat ini
     * Point titikB : titik sebelumnya
     *
     */
    private List<Integer> hapusTitik(List<MatOfPoint> contours, Mat hand, ArrayList<MatOfPoint> contour) {
        List<Integer> puncak = new ArrayList<>();
        try {

            Point[] point = contour.get(0).toArray();
//            puncak.addAll(devContourIdxList);
//            lembah.addAll(devContourIdxList);
            //jika posisi false berarti cari lembah
            //jika posisi true berarti cari puncak
            Boolean isPuncak = true;
            for (int j = 0; j < puncak.size(); j++) {
                int index = puncak.get(j);
                int indexP = 0;
                if (j + 1 < puncak.size()) {
                    indexP = puncak.get(j + 1);
                }

                if (index < point.length && indexP < point.length
                        && point[index].y
                        < hand.rows() - 1
                        && point[indexP].y < hand.rows() - 1) {
                    if (isPuncak) {
                        //jika menemukan puncak index dicaatat
                        if (Preprocessing.arahTitikY(point[index], point[indexP])) {
                            isPuncak = false;
//                            lembah.set(j, -1);
                        } //jika titik lebih tinggi index sebelumnya dihapus
                        else {
//                            lembah.set(j, -1);
                            puncak.set(j, -1);
//                            devContourIdxList.set(j, -1);
                        }
                    } else {
                        //jika menemukan lembah index dicaatat
                        if (Preprocessing.arahTitikY(point[indexP], point[index])) {
                            puncak.set(j, -1);
                            isPuncak = true;
                        } //jika titik lebih tinggi index sebelumnya dihapus
                        else {
//                            lembah.set(j, -1);
                            puncak.set(j, -1);
//                            devContourIdxList.set(j, -1);
                        }
                    }
                } else {
                    puncak.set(j, -1);
//                    lembah.set(j, -1);
//                    devContourIdxList.set(j, -1);
                }
            }
            Integer rem = -1;
//            puncak.addAll(lembah);
            Collections.sort(puncak);

            for (Integer integer : puncak) {
                System.out.println(integer);
            }
            System.out.println("");
            System.out.println("contous " + contour.get(0).toArray().length);
            while (puncak.contains(rem)) {
                puncak.remove(rem);
            }
            System.out.println("");
            for (Integer integer : puncak) {
                System.out.println(integer);
            }
            Mat hand2 = hand.clone();
            Preprocessing.drawPointColor(contour, hand2, puncak);
            Preprocessing.drawJumlahJari(hand, puncak.size());
//            layarBW.setImage(Utils.mat2Image(hand2));
//            drawPointColor(contous, hand, devContourIdxList);
            layarEdge.setImage(Utils.mat2Image(hand2));

        } catch (Exception e) {
            System.out.
                    println("hapusTitik(List<MatOfPoint> contours, Mat hand)");
            System.out.println(e);
            System.out.println("");
        }
        return puncak;
    }

//######################################################################
    /**
     * method layarMain OnClick
     * memperoleh nilai koordinat layarMain
     * var:
     * TextField txtV : text field menyimpan koordinat X
     * TextField txtS : text field menyimpan koordinat Y
     *
     */
    @FXML
    private void getPoint(MouseEvent event) {
        txtMainFramePoint.setText("(" + String.valueOf(event.getX()) + ", " + String.valueOf(event.getY()) + ")");
    }

}
