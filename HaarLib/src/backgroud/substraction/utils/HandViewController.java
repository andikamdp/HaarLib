/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the templateasdasdasd in the editor.
 */
package backgroud.substraction.utils;

import java.net.URL;
import java.util.ArrayList;
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
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.TrainData;
import org.opencv.videoio.VideoCapture;

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
    private ScheduledExecutorService timer;
    private VideoCapture capture;
    private boolean cameraActive;
    private double aWeight;
    private int top, right, bottom, left;
    private int num_frame;

    /**
     * Initializes the controller class.
     */
    @Override
    public void initialize(URL url, ResourceBundle rb) {
        // TODO
        capture = new VideoCapture();
        cameraActive = false;
        aWeight = 0.5;
        top = 10;
        right = 350;
        bottom = 225;
        left = 590;
        num_frame = 0;

    }

    @FXML
    private void StartCameraOnClick(ActionEvent event) {
        if (!cameraActive) {
            capture.open(0);
            if (this.capture.isOpened()) {
                cameraActive = true;

                Runnable frameGrabber = new Runnable() {
                    @Override
                    public void run() {
                        Mat frame = grabFrame();
                        //# flip the frame so that it is not the mirror view
                        Core.flip(frame, frame, 1);
                        //# clone the frame// belum dipakai
                        Mat roi;
                        //# get the height and width of the frame
                        Size s = frame.size();
                        //# get the ROI
//                        Rect rectCrop = new Rect(frame.cols() - 50, frame.rows()
//                                - 50,
//                                frame.width() - 50, frame.height() - 50);
//                        Rect rectCrop = new Rect(235, 235, 235, 235);
                        //frame = frame.submat(rectCrop);
                        //# convert the roi to grayscale and blur it
                        Imgproc.GaussianBlur(frame, frame, new Size(7, 7), 0);

                        //Imgproc.accumulateWeighted(frame, frame, 0.5);
                        frame = segment(frame);

                        Image imageToMat = Utils.mat2Image(frame);
                        updateImageView(layarMain, imageToMat);

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

    @FXML
    private void UpdateCameraOnClick(ActionEvent event) {
        Imgcodecs.imwrite("E:\\TA\\opencv.jpg", grabFrame());
    }
/////////////

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

    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

    /**
     * On application close, stop the acquisition from the camera
     */
    protected void setClosed() {
        this.stopAcquisition();
    }

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
                    ///////////Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
//                    Imgproc.accumulateWeighted(frame, frame, 50);
                }

            } catch (Exception e) {
                // log the error
                System.err.println("Exception during the image elaboration: "
                        + e);
            }
        }
        return frame;
    }

    /**
     * Stop the acquisition from the camera and release all the resources
     */
    public Image bg;
    public Mat diff, diff2, treshold;
//method untuk memisahkan objek dengan background

    private Mat segment(Mat frame) {
        double tres = 50.0;
        Mat frame2 = frame.clone();
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY);
        diff = Imgcodecs.imread("E:\\TA\\opencv.jpg");
///////////        Imgproc.cvtColor(diff, diff, Imgproc.COLOR_BGR2GRAY);
//        diff = Imgcodecs.imread("E:\\TA\\opencv-test.png");
//        diff2 = Imgcodecs.imread("E:\\TA\\opencv-logo.jpg");
//
        Imgproc.cvtColor(diff, diff, Imgproc.COLOR_BGR2GRAY);
        Core.flip(diff, diff, 1);
        //layarBW.setImage(Utils.mat2Image(diff));
//        Core.convertScaleAbs(frame, diff);
        Mat dist = new Mat();

        Core.absdiff(diff, frame, dist);
//        Core.absdiff(diff, diff2, dist);
//batas minimum treshold
        layarEdge.setImage(Utils.mat2Image(dist));

        Imgproc.threshold(dist, frame, tres, 255, Imgproc.THRESH_BINARY);
        ///
        Mat hierarchy = new Mat();
        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(frame.clone(), contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (hierarchy.size().height > 0 && hierarchy.size().width > 0) {
            // for each contour, display it in blue
            for (int idx = 0; idx >= 0; idx = (int) hierarchy.get(0, idx)[0]) {
                Imgproc.
                        drawContours(frame2, contours, idx,
                                new Scalar(250, 0, 0), 3);
            }
        }
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BayerBG2BGR);
        layarEdge.setImage(Utils.mat2Image(frame));
//        layarBW.setImage(Utils.mat2Image(contours.get(0)));
        Mat c = contours.get(0);

//        for (int i = 0; i < c.cols(); i++) {
//            for (int j = 0; j < c.rows(); j++) {
//                System.out.print(c.get(i, j)[0]);
//
//            }
//            System.out.println("");
//        }
        return frame2;
    }
//method untuk deteksi tangan

    private void HandRec(Mat thresholded, MatOfPoint segmented) {

        //find the convex hull of the segmented hand region
        MatOfInt chull = new MatOfInt();
        Imgproc.convexHull(segmented, chull);

//find the most extreme points in the convex hull
//      #extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
//	#extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
//	#extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
//	#extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])
        double[] extreme_top = chull.get(0, 0);
        double[] extreme_buttom = chull.get(chull.rows() - 1, 0);
        double[] extreme_right = chull.get(0, chull.cols() - 1);
        double[] extreme_left = chull.get(0, chull.cols() - 1);

//	# find the center of the palm
//	cX = (extreme_left[0] + extreme_right[0]) / 2
//	cY = (extreme_top[1] + extreme_bottom[1]) / 2
        //    int cX = (extreme_left[0] + extreme_right[0]) / 2;
    }

//	# find the maximum euclidean distance between the center of the palm
//	# and the most extreme points of the convex hull
//	distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
//	maximum_distance = distance[distance.argmax()]
//
//	# calculate the radius of the circle with 80% of the max euclidean distance obtained
//	radius = int (
//
//
//0.8 * maximum_distance)
//
//	# find the circumference of the circle
//	circumference = (2 * np.pi * radius)
//
//	# take out the circular region of interest which has
//	# the palm and the fingers
//	circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
//
//	# draw the circular ROI
//	cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
//
//	# take bit-wise AND between thresholded hand using the circular ROI as the mask
//	# which gives the cuts obtained using mask on the thresholded hand image
//	circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
//
//	# compute the contours in the circular ROI
//	(_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
//
//	# initalize the finger count
//	count = 0
//
//	# loop through the contours found
//	for c in cnts:
//		# compute the bounding box of the contour
//		(x, y, w, h) = cv2.boundingRect(c)
//
//		# increment the count of fingers only if -
//		# 1. The contour region is not the wrist (bottom area)
//		# 2. The number of points along the contour does not exceed
//		#     25% of the circumference of the circular ROI
//		if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
//			count += 1
//
//	return count
}
