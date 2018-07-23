/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package backgroud.substraction.utils;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;
import org.opencv.core.Core;

/**
 *
 * @author Andika Mulyawan
 */
public class MainApp extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        FXMLLoader loader = new FXMLLoader();
        loader.setLocation(MainApp.class.getResource("HandView.fxml"));
        AnchorPane root = loader.load();
        Scene scane = new Scene(root);

        primaryStage.setScene(scane);
        primaryStage.show();
    }

}
