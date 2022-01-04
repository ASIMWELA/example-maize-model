
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

@Slf4j
public class MaizeDiseaseDetectionModel {

    public static void main(String[] args) throws IOException {
        String BASE_PATH = "Dataset";
        int height = 150;    // height of the picture in px
        int width = 150;     // width of the picture in px
        int channels = 3;   // 3 channels for colored  images
        //TODO: Update to number of output of maize diseases
        int outputNum = 4; // 10 digits classification
        int batchSize = 54; // number of samples that will be propagated through the network in each iteration
        int nEpochs = 1;    // number of training epochs

        int seed = 1234;    // number used to initialize a pseudorandom number generator.
        Random randNumGen = new Random(seed);
        double learningRate = 0.0015;

        //TODO: direct to maize diseases data
        File trainData = new File(BASE_PATH + "/train");
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // use parent directory name as the image label
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        trainRR.initialize(trainSplit);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

        // pixel values from 0-255 to 0-1 (min-max scaling)
        DataNormalization imageScaler = new ImagePreProcessingScaler();
        imageScaler.fit(trainIter);
        trainIter.setPreProcessor(imageScaler);

        // vectorization of test data
        //TODO: direct to maize diseases data
        File testData = new File(BASE_PATH + "/test");
        FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testSplit);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
        testIter.setPreProcessor(imageScaler); // same normalization for better results

        log.info("Network configuration and training...");
        // reduce the learning rate as the number of training epochs increases
        // iteration #, learning rate
//        Map<Integer, Double> learningRateSchedule = new HashMap<Integer, Double>();
//        learningRateSchedule.put(0, 0.06);
//        learningRateSchedule.put(200, 0.05);
//        learningRateSchedule.put(600, 0.028);
//        learningRateSchedule.put(800, 0.0060);
//        learningRateSchedule.put(1000, 0.001);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2( 0.001) // ridge regression value
                .updater(new Nadam())
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1) // nIn need not specified in later layers
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(height, width, channels)) // InputType.convolutional for normal image
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10));
        log.info("Total num of params: {}", net.numParams());

        // evaluation while training (the score should go down)
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainIter);
            log.info("Completed epoch {}", i);
            Evaluation eval = net.evaluate(testIter);
            log.info(eval.stats());

            trainIter.reset();
            testIter.reset();
        }

        File maizeModelPath = new File(BASE_PATH + "/maize-disease-model.zip");
        ModelSerializer.writeModel(net, maizeModelPath, true);
        log.info("The Maize disease detetion model has been saved in {}", maizeModelPath.getPath());
    }
}
