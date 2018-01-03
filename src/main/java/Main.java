/*
Mohammed Alhakeem
AIM3-HW2-Q5-a
*/

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.*;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        //list for storing the centroids
        List<String> centroids = new ArrayList<String>();

        //configuring spark
        SparkConf conf = new SparkConf().setAppName("cust data").setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        // Load and parse data
        String path = "src/main/resources/mnist_test.csv";
        JavaRDD<String> data = jsc.textFile(path);
        JavaRDD<Vector> parsedData = data.map(s -> {
            String[] sarray = s.split(",");
            double[] values = new double[sarray.length];
            for (int i = 0; i < sarray.length; i++) {
                values[i] = Double.parseDouble(sarray[i]);
            }
            return Vectors.dense(values);
        });
        parsedData.cache();

        // Cluster the data into 10 classes using KMeans with 150 iterations
        int numClusters = 10;
        int numIterations = 150;
        KMeansModel clusters = KMeans.train(parsedData.rdd(), numClusters, numIterations);

        System.out.println("Cluster centers:");

        for (Vector center : clusters.clusterCenters()) {
            System.out.println("center: " + center);
            //adding each centroid to the list
            centroids.add(center.toString());
        }

        //writing the centroids to a csv file
        writeCSV(centroids);

        double cost = clusters.computeCost(parsedData.rdd());
        System.out.println("Cost: " + cost);

        // Evaluate clustering by computing Within Set Sum of Squared Errors
        double WSSSE = clusters.computeCost(parsedData.rdd());
        System.out.println("Within Set Sum of Squared Errors = " + WSSSE);

    }

    public static void writeCSV(List l) {
        Path file = Paths.get("src/main/resources/centers.csv");
        try {
            Files.write(file, l, Charset.forName("UTF-8"));
            System.out.println("done");
        } catch (IOException e) {
            System.out.println("Failed!");
        }
    }
}