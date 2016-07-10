package com.rc;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.stream.Collectors;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LSTM extends Model {

	private static Logger log = LoggerFactory.getLogger(LSTM.class);
	private Random random = new Random( 300 ) ;

	public LSTM( Path configDir ) throws IOException {
		super( configDir ) ;
	}

	public LSTM() {
		super() ;
	}


	protected Collection<DataSet> getDatasets( Path source ) throws IOException {

		this.batchSize = 50 ;

		log.info("Load data from " + source );

		List<String[]> cols = Files.lines( source ) 
				.map( s -> s.split(",") )
				.collect( Collectors.toList() )
				;

		List<DataSet> dsets = new ArrayList<>() ;

		int timeSeriesLength = ( cols.get(0).length / numInputs ) - 1 ;

		int numRowsProcessed = 0 ;
		while( numRowsProcessed<cols.size() ) {

			int batchSize = Math.min( this.batchSize, (cols.size()-numRowsProcessed) ) ;

			INDArray input  = Nd4j.create(batchSize, numInputs, timeSeriesLength ) ;
			INDArray labels = Nd4j.create(batchSize, numOutputs, timeSeriesLength ) ;

			int ix[] = new int[3] ;
			for( int i=0 ; i<batchSize ; i++ ) {
				ix[0] = i ;
				String row[] = cols.get(numRowsProcessed) ;
				numRowsProcessed++ ;

				for( int j=0 ; j<timeSeriesLength ; j++ ) {
					ix[2] = j ;

					for( int k=0 ; k<numInputs ; k++ ) {
						ix[1] = k ;
						float n = Float.parseFloat( row[j*numInputs + k] ) ;
						input.putScalar( ix, n ) ;
					}
					for( int k=0 ; k<numOutputs ; k++ ) {
						ix[1] = k ;
						float n = Float.parseFloat( row[(j+1)*numInputs + k] ) ;
						labels.putScalar( ix, n ) ;
					}
				}
			}
			dsets.add( new DataSet(input,labels) ) ;
		}
		return dsets ;
	}

	@Override
	public BlockingQueue<String> train( Path trainingData ) throws Exception {

		StreamIterationListener sil = new StreamIterationListener(100) ;
		forEach( mln -> mln.setListeners(sil) ) ;

		Collection<DataSet> dsets = getDatasets( trainingData ) ;

		Runnable r = new Runnable() {
			@Override
			public void run() {
				for( int epoch=0 ; epoch<100 ; epoch++ ) {
					log.info("Train model - epoch {}", epoch );

					for( DataSet ds : dsets ) { 
						getModel( 0 ).fit( ds ) ;
					}		
				}
				log.info("Training done.");
			}
		} ;
		new Thread( r ).start(); 
		return sil.getStream() ;
	}


	@Override
	public String test(Path testData) throws Exception {

		int numTests = 10 ;
		int timeSeriesLength = 1 ;
		StringBuilder rc = new StringBuilder() ;

		for( int t=0 ; t<numTests ; t++ ) {

			INDArray input  = Nd4j.create(1, numInputs, timeSeriesLength ) ;

			int ix[] = new int[3] ;
			ix[0] = 0 ;
			double n = random.nextDouble() ;

			ix[2] = 0 ;
			for( int k=0 ; k<numInputs ; k++ ) {
				ix[1] = k ;
				input.putScalar( ix, n ) ;
			}

			getModel(0).rnnClearPreviousState();
			INDArray p1 = getModel(0).rnnTimeStep(input);
			INDArray p2 = getModel(0).rnnTimeStep(input);
			INDArray p3 = getModel(0).rnnTimeStep(input);
			rc.append( "Input =" ).append( n ).append( " Output=").append( p1.getDouble( 0,0,0 ) ).append(',').append( p2.getDouble( 0,0,0 ) ).append(',').append( p3.getDouble( 0,0,0 ) ).append( "\n" );
		}
		log.info("All Done");

		return rc.toString() ;
	}

	public void createModelConfig( int numLayers, int numInputs, int numOutputs ) {
		this.numInputs = numInputs ;
		this.numOutputs = numOutputs ;

		ListBuilder lb = new NeuralNetConfiguration.Builder()
				.seed( 100 )
				.iterations( 1 )
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.learningRate(0.1)
				.rmsDecay(0.95)
				.regularization(true)
				.l2(0.001)
				.weightInit(WeightInit.XAVIER)
				.updater(Updater.RMSPROP)
				.list(numLayers)
				;

		for( int i=0 ; i<numLayers-1 ; i++ ) {
			lb.layer(i, new GravesLSTM.Builder()
					.nIn(numInputs)
					.nOut(numInputs)
					.activation("relu")
					.build()
					) ;
		}

		lb.layer( numLayers-1, new RnnOutputLayer.Builder()
				.activation("relu")
				.lossFunction(LossFunctions.LossFunction.MCXENT)
				.nIn(numInputs)
				.nOut(numInputs)
				.build()
				) ;

		MultiLayerConfiguration conf = lb
				.backprop(true)
				.backpropType(BackpropType.TruncatedBPTT)
				.tBPTTForwardLength(4)
				.tBPTTBackwardLength(4)
				.pretrain(false)
				.build();

		addModel(conf);  
	}
}

