/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the templateasdasdasd in the editor.
 */
package src.controller;

import src.Utils;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.ResourceBundle;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.SVM;
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
    private Button btnStartCamera;
    @FXML
    private Button btnUpdateCamera;
    @FXML
    private TextField txtH;
    @FXML
    private TextField txtV;
    @FXML
    private TextField txtS;
    @FXML
    private TextField txtValue;
    @FXML
    private ImageView layarMain;
//
    private MainAppController mainAppController;
    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    int i = 0;
    SVM s;
//    List<Integer> devContourIdxList;
//    List<MatOfPoint> contous;
//    List<MatOfInt4> devOfInt4s;
//    List<MatOfPoint> devOfPoints;
//    List<Integer> puncak = new ArrayList<>();
//    List<Integer> lembah = new ArrayList<>();

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        capture = new VideoCapture();
        cameraActive = false;
        i = 0;

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
//        s = SVM.load("E:\\TA\\hCoba.xml");
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
                            System.out.println(e);
                        }
//                        s.predict(frame);
//                        Start(frame);
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
        Image imageToMat;
        Core.flip(frame, frame, 1);
        Preprocessing.drawRect(frame);
        Mat hand = Preprocessing.getBox(frame.clone());
        Mat tresholded;
        if (txtS.getText().isEmpty()) {
            tresholded = Preprocessing.segment(hand.clone(), Double.valueOf(txtValue.getText()));
        } else {
            tresholded = Preprocessing.segmentInvers(hand.clone(), Double.valueOf(txtValue.getText()));
        }
        imageToMat = Utils.mat2Image(tresholded);
        updateImageView(layarBW, imageToMat);
        List<MatOfPoint> contour = Preprocessing.getContour(tresholded);
        List<MatOfInt4> devOfInt4s = Preprocessing.getDevectIndexPoint(contour);
//        Preprocessing.toListMatOfPointDevec(contour, devOfInt4s, devContourIdxList);
        List<Point> pointContourSorted = Preprocessing.toListContour(contour.get(0));
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
        //

//        Mat handView = Preprocessing.drawRect(hand.clone(), p, p_);
        Mat handView = Preprocessing.getEdge_2(hand.clone());
//        hapusTitik(contous, Preprocessing.getBox(frame));
        layarMain.setImage(Utils.mat2Image(frame));
        imageToMat = Utils.mat2Image(tresholded);
        updateImageView(layarBW, imageToMat);
        imageToMat = Utils.mat2Image(handView);
        updateImageView(layarEdge, imageToMat);
        hand = Preprocessing.getBox(hand, p, p_);
        System.out.println(handView.get(0, 0).length);
//        captureImage(hand);
        for (int j = 0; j < handView.rows(); j++) {
            for (int k = 0; k < handView.cols(); k++) {
                System.out.print(handView.get(j, k)[0] + " ");
            }
            System.out.println("");
        }
        System.out.println("p");
        handView = Preprocessing.getEdge(hand.clone());
        for (int j = 0; j < handView.rows(); j++) {
            for (int k = 0; k < handView.cols(); k++) {
                System.out.print(handView.get(j, k)[0] + " ");
            }
            System.out.println("");
        }
        System.out.println("q");

    }

//######################################################################
    /**
     * method button btnUpdateCamera OnClick
     * var:
     * int i : reset nomor urut gambar yang di simpan
     */
    @FXML
    private void updateCameraOnClick(ActionEvent event) {
        //mengambil gambar background
//        Imgcodecs.imwrite("E:\\TA\\opencv.jpg", grabFrame());
//        captureImage();
//        imwrite_DELETE();
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
        Imgcodecs.imwrite("E:\\TA\\HandLearnSVM\\BISINDO\\" + txtH.getText() + "\\" + txtV.getText() + "_" + i
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
    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
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
//        String bg = "E:\\TA\\h0.jpg";
        Mat hand;
        if (txtH.getText().isEmpty()) {
            hand = Imgcodecs.imread("C:\\Users\\Andika Mulyawan\\Desktop\\1.jpg");
        } else {
            hand = Imgcodecs.imread("E:\\TA\\HandLearnSVM\\penuh\\hfull" + txtH.getText() + ".jpg");
        }
        start(hand);
//        Image edge = layarEdge.getImage();

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
        txtV.setText(String.valueOf(event.getX()));
        txtS.setText(String.valueOf(event.getY()));
    }

}
