package net.sf.jclal.examples;

import net.sf.jclal.activelearning.algorithm.ClassicalALAlgorithm;
import net.sf.jclal.activelearning.batchmode.QBestBatchMode;
import net.sf.jclal.activelearning.oracle.SimulatedOracle;
import net.sf.jclal.activelearning.scenario.PoolBasedSamplingScenario;
import net.sf.jclal.activelearning.singlelabel.querystrategy.WithUnknownClassQueryStrategy;
import net.sf.jclal.activelearning.stopcriteria.SurpriseBased;
import net.sf.jclal.activelearning.stopcriteria.UnlabeledSetEmpty;
import net.sf.jclal.classifier.WekaClassifier;
import net.sf.jclal.core.IQueryStrategy;
import net.sf.jclal.evaluation.method.kFoldCrossValidation;
import net.sf.jclal.listener.GraphicalReporterListener;
import net.sf.jclal.sampling.supervised.Resample;
import net.sf.jclal.util.random.RanecuFactory;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMOsync;

/**
 * Example of K fold cross validation experiment that uses the unknown class query strategy
 * as query strategy, pool-base as scenario and the classical AL process.
 *
 * @author Oscar Gabriel Reyes Pupo
 * @author Eduardo Perez Perdomo
 */
public class SurprisedBasedActiveLearningWithUnknownClassesExample {

    public static void main(String[] args) {

        String fileName = "datasets/diabetes_arff.arff";

        // The initial labeled set from the training set is randomly
        // selected
        Resample sampling = new Resample();

        sampling.setNoReplacement(false);
        sampling.setInvertSelection(false);

        sampling.setPercentageInstancesToLabelled(0.15);

        // Set the scenario to use
        PoolBasedSamplingScenario scenario = new PoolBasedSamplingScenario();
        QBestBatchMode batchMode = new QBestBatchMode();
        batchMode.setBatchSize(1);

        scenario.setBatchMode(batchMode);

        //Set the oracle
        SimulatedOracle oracle = new SimulatedOracle();

        scenario.setOracle(oracle);

        // Set the query strategy to use
        IQueryStrategy queryStrategy = new WithUnknownClassQueryStrategy();

        // Set the base classifier to use in the query strategy
        WekaClassifier model = new WekaClassifier();

        Classifier classifier = new SMOsync();

        model.setClassifier(classifier);

        //Set the model into the query strategy
        queryStrategy.setClassifier(model);

        //Set the query strategy into the scenario
        scenario.setQueryStrategy(queryStrategy);

        // Set the algorithm's listeners
        GraphicalReporterListener visual = new GraphicalReporterListener();
        visual.setReportOnFile(true);
        visual.setReportTitle("margin");
        visual.setShowSeparateWindow(true);
        visual.setReportFrequency(1);
        visual.setReportDirectory("reports/ecoli");

        // Construct the Active Learning algorithm
        ClassicalALAlgorithm algorithm = new ClassicalALAlgorithm();

        //Set the listener for the algorithm
        algorithm.addListener(visual);

        //Set the stop criteria
        SurpriseBased stop1= new SurpriseBased();
        stop1.setMinSurprise(0.05);
        stop1.setInrConverge(15);

        UnlabeledSetEmpty stop2= new UnlabeledSetEmpty();

        algorithm.addStopCriterion(stop1);
        algorithm.addStopCriterion(stop2);

        // Set the scenario into the algorithm
        algorithm.setScenario(scenario);

        //Set the evaluation method to use
        kFoldCrossValidation method = new kFoldCrossValidation();

        //Set the sampling strategy into the algorithm
        method.setSamplingStrategy(sampling);

        //Set the path of the dataset
        method.setFileDataset(fileName);

        method.setNumFolds(10);
        method.setStratify(true);

        RanecuFactory random = new RanecuFactory();
        // random.setSeed((int)System.currentTimeMillis());
        random.setSeed(0);

        method.setRandGenFactory(random);

        //Set the algorithm into the evaluation method
        method.setAlgorithm(algorithm);
        method.evaluate();
    }

}
