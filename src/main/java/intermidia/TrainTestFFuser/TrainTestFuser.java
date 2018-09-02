package intermidia.TrainTestFFuser;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

import org.openimaj.math.statistics.distribution.Histogram;
import org.openimaj.ml.clustering.DoubleCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.DoubleKMeans;
import org.openimaj.util.pair.IntDoublePair;

import com.opencsv.CSVReader;

public class TrainTestFuser 
{
	private static int k;
	private static int clusteringSteps;
	private final static int boostFactor = 2;
	
	
	//Usage: <in: first bof file> ... <in: last bof file> <out: fused bof file> <in: k> <in: clustering steps>
	//<in: pooling mode> <in: normalisation> 
    public static void main( String[] args ) throws Exception
    {
    	ArrayList<ArrayList<Histogram>> trainFeaturesArrays = new ArrayList<ArrayList<Histogram>>();
    	ArrayList<ArrayList<Histogram>> testFeaturesArrays = new ArrayList<ArrayList<Histogram>>();
    	int featureTypes = 0;
    	int featureWords = 0;
    	int trainVideoQty = 0;
    	int testVideoQty = 0;
    	
    	/*Set k and maximum clustering steps*/
    	k = Integer.parseInt(args[args.length - 4]);
    	clusteringSteps = Integer.parseInt(args[args.length - 3]);
    	String poolingMode = args[args.length - 2];
    	boolean normalise = args[args.length - 1].equals("normalise");
    	   	
    	//Read different feature word histogram files - TRAIN SET
    	for(featureTypes = 0; featureTypes < ((args.length - 6)/2); featureTypes++)
    	{
    		CSVReader featureReader = new CSVReader(new FileReader(args[featureTypes]), ' ');
    		String[] line;
    		trainFeaturesArrays.add(new ArrayList<Histogram>());
    		double maxValue = Double.MIN_VALUE;
    		while ((line = featureReader.readNext()) != null) 
    		{		
    			int fvSize = line.length - 1;    			
    			double fv[] = new double[fvSize];
    			for(int j = 0; j < fvSize; j++)
    			{
    				fv[j] = Double.parseDouble(line[j + 1]);
    				if(fv[j] > maxValue)
    				{
    					maxValue = fv[j];
    				}   
    			}
    			trainFeaturesArrays.get(featureTypes).add(new Histogram(fv));    			
    		}
    		
    		//Normalize values   		
    		if(normalise)
    		{
    			for(int i = 0; i < trainFeaturesArrays.get(featureTypes).size(); i++)
    			{
    				
    				for(int j = 0; j < trainFeaturesArrays.get(featureTypes).get(i).length(); j++)
    				{
    					double normalisedValue = trainFeaturesArrays.get(featureTypes).get(i).get(j) / maxValue * 100;
    					trainFeaturesArrays.get(featureTypes).get(i).setFromDouble(j, normalisedValue);
    				}
    			}
    		}    		    		

    		//Sums all feature vector lengths for all different types
    		featureWords += trainFeaturesArrays.get(featureTypes).get(0).length();    		    		
    		featureReader.close();
    	}
    	
    	//Read different feature word histogram files - TEST SET
    	for(featureTypes = 0; featureTypes < ((args.length - 6)/2); featureTypes++)
    	{
    		CSVReader featureReader = new CSVReader(new FileReader(args[featureTypes + ((args.length - 6)/2)]), ' ');
    		String[] line;
    		testFeaturesArrays.add(new ArrayList<Histogram>());
    		double maxValue = Double.MIN_VALUE;
    		while ((line = featureReader.readNext()) != null) 
    		{		
    			int fvSize = line.length - 1;    			
    			double fv[] = new double[fvSize];
    			for(int j = 0; j < fvSize; j++)
    			{
    				fv[j] = Double.parseDouble(line[j + 1]);
    				if(fv[j] > maxValue)
    				{
    					maxValue = fv[j];
    				}   
    			}
    			testFeaturesArrays.get(featureTypes).add(new Histogram(fv));    			
    		}
    		
    		//Normalize values   		
    		if(normalise)
    		{
    			for(int i = 0; i < testFeaturesArrays.get(featureTypes).size(); i++)
    			{
    				
    				for(int j = 0; j < testFeaturesArrays.get(featureTypes).get(i).length(); j++)
    				{
    					double normalisedValue = testFeaturesArrays.get(featureTypes).get(i).get(j) / maxValue * 100;
    					testFeaturesArrays.get(featureTypes).get(i).setFromDouble(j, normalisedValue);
    				}
    			}
    		}    		    		
    		featureReader.close();
    	}  
    	
    	
    	//Relate each TRAIN word with its modality
    	int insertedTrainFeatureSum = 0;
    	ArrayList<Integer> trainWordModality = new ArrayList<Integer>();
    	for(int i = 0; i < featureTypes; i++)
    	{
    		int modalityLength = trainFeaturesArrays.get(i).get(0).length();
    		for(int j = 0; j < modalityLength; j++)
    		{
    			trainWordModality.add(i);
    		}
    		insertedTrainFeatureSum += modalityLength;
    	}
    	//Relate each TEST word with its modality
    	int insertedTestFeatureSum = 0;
    	ArrayList<Integer> testWordModality = new ArrayList<Integer>();
    	for(int i = 0; i < featureTypes; i++)
    	{
    		int modalityLength = testFeaturesArrays.get(i).get(0).length();
    		for(int j = 0; j < modalityLength; j++)
    		{
    			testWordModality.add(i);
    		}
    		insertedTestFeatureSum += modalityLength;
    	} 	

    	
    	//Compute the TRAIN transpose matrix
    	trainVideoQty = trainFeaturesArrays.get(0).size();
    	double[][] trainFeaturePool = new double[featureWords][trainVideoQty];
    	insertedTrainFeatureSum = 0;    	
    	for(ArrayList<Histogram> arrayList: trainFeaturesArrays)
    	{
    		int trainVideo = 0;
    		for(Histogram histogram: arrayList)
    		{
    			for(int j = 0; j < histogram.getVector().length; j++)
    			{
    				trainFeaturePool[insertedTrainFeatureSum + j][trainVideo] = histogram.get(j);
    			}
    			trainVideo++;
    		}
    		//Sums already processed feature words
    		insertedTrainFeatureSum += arrayList.get(0).getVector().length;
    	}
    	//Compute the TEST transpose matrix
    	testVideoQty = testFeaturesArrays.get(0).size();
    	double[][] testFeaturePool = new double[featureWords][testVideoQty];
    	insertedTestFeatureSum = 0;    	
    	for(ArrayList<Histogram> arrayList: testFeaturesArrays)
    	{
    		int testVideo = 0;
    		for(Histogram histogram: arrayList)
    		{
    			for(int j = 0; j < histogram.getVector().length; j++)
    			{
    				testFeaturePool[insertedTestFeatureSum + j][testVideo] = histogram.get(j);
    			}
    			testVideo++;
    		}
    		//Sums already processed feature words
    		insertedTestFeatureSum += arrayList.get(0).getVector().length;
    	}
    	
    	//Performs clustering of the TRAIN feature words by their video histograms
    	DoubleKMeans clusterer = DoubleKMeans.createExact(k, clusteringSteps);
    	DoubleCentroidsResult centroids = clusterer.cluster(trainFeaturePool);
    	HardAssigner<double[], double[], IntDoublePair> hardAssigner = centroids.defaultHardAssigner();
    	
    	//Assign each feature word to a cluster that correspond a multimodal feature
    	ArrayList<ArrayList<Integer>> featureGroups = new ArrayList<ArrayList<Integer>>();
    	for(int i = 0; i < k; i++)
    	{
    		featureGroups.add(new ArrayList<Integer>());
    	}
    	
    	for(int featureIndex = 0; featureIndex < featureWords; featureIndex++)
     	{

    		int group = hardAssigner.assign(trainFeaturePool[featureIndex]) ;
     		featureGroups.get(group).add(featureIndex);
     	}
    	
    	//Compute train/test set multimodal words histograms by average pooling    	
    	double[][] hTrain;
    	double[][] hTest;
    	switch(poolingMode)
    	{
    		//Jhuo et al. Average Pooling Strategy 
    		case "javg":
    		{    		
    			hTrain = jhuoAveragePooling(featureGroups, trainWordModality, featureTypes, trainFeaturePool, trainVideoQty, k);
    			hTest = jhuoAveragePooling(featureGroups, testWordModality, featureTypes, testFeaturePool, testVideoQty, k);
    			break;
    		}
    		//Standard Max Pooling
    		case "max":
    		{
    			hTrain = maxPooling(featureGroups, trainFeaturePool, trainVideoQty, k, false);
    			hTest = maxPooling(featureGroups, testFeaturePool, testVideoQty, k, false);
    			break;
    		}
    		//Standard Average Pooling
    		case "avg":
    		{
    			hTrain = averagePooling(featureGroups, trainFeaturePool, trainVideoQty, k, false);
    			hTest = averagePooling(featureGroups, testFeaturePool, testVideoQty, k, false);
    			break;
    		}
    		//Co-occurrence Boosted Max Pooling
    		case "cobmax":
    		{
    			hTrain = maxPooling(featureGroups, trainFeaturePool, trainVideoQty, k, true);
    			hTest = maxPooling(featureGroups, testFeaturePool, testVideoQty, k, true);
    			break;
    		}
    		//Co-occurrence Boosted Average Pooling
    		case "cobavg":
    		{
    			hTrain = averagePooling(featureGroups, trainFeaturePool, trainVideoQty, k, true);
    			hTest = averagePooling(featureGroups, testFeaturePool, testVideoQty, k, true);
    			break;
    		}    			    		    		
    		default:
    		{
    			hTrain = averagePooling(featureGroups, trainFeaturePool, trainVideoQty, k, false);
    			hTest = averagePooling(featureGroups, testFeaturePool, testVideoQty, k, false);
    			break;
    		}
    	}
    	
    	//Write train features on output 
    	FileWriter outputTrain = new FileWriter(args[args.length - 6]);
    	for(int i = 0; i < trainVideoQty; i++)
    	{	
    		//Shot number
    		outputTrain.write(i + " ");
    		for(int j = 0; j < k; j++)
    		{
    			if(j < (k - 1))
    			{
   					outputTrain.write(hTrain[i][j] + " ");    					
    			}else
    			{
    				outputTrain.write(hTrain[i][j] + "\n");    					
    			}
    		}    		
    	}    	
    	outputTrain.close();

    	//Write test features on output 
    	FileWriter outputTest = new FileWriter(args[args.length - 5]);
    	for(int i = 0; i < testVideoQty; i++)
    	{	
    		//Shot number
    		outputTest.write(i + " ");
    		for(int j = 0; j < k; j++)
    		{
    			if(j < (k - 1))
    			{
   					outputTest.write(hTest[i][j] + " ");    					
    			}else
    			{
    				outputTest.write(hTest[i][j] + "\n");    					
    			}
    		}    		
    	}    	
    	outputTest.close();
    	
   }

    private static double[][] jhuoAveragePooling(ArrayList<ArrayList<Integer>> featureGroups, ArrayList<Integer> vectorModality, int vectorModalities,  double featurePool[][], int shots, int k)
    {   	    	
    	double[][] hJhuoAvg = new double[shots][k]; 
    	double[][] doubleHJhuoAvg = new double[shots][k];
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{    			  			    			
    			/*Divide over different modalities*/    			
    			@SuppressWarnings("unchecked")
    			ArrayList<Integer> words[] = new ArrayList[vectorModalities];
    			for(int l = 0; l < vectorModalities; l++)
    			{
    				words[l] = new ArrayList<Integer>();
    			}
    			for(Integer val: featureGroups.get(j))
    			{
    				words[vectorModality.get(val)].add(val);
    			}  
   			    		
    			/*Sum pairwise individual modalities weights*/
    			double sum = 0;
    			//Iterate each group modalities
    			for(int l = 0; l < vectorModalities; l++)
    			{
    				//Iterate each element of each modality
    				for(int m = 0; m < words[l].size(); m++)
    				{
    					//For each element, combine with other, avoiding repeated combinations
    					for(int n = l + 1; n < vectorModalities; n++)
    					{
    						for(int o = 0; o < words[n].size(); o++)
    						{   							
    							sum += (featurePool[words[l].get(m)][i] + featurePool[words[n].get(o)][i]);
    						}
    					}
    				}
    			} 
    			hJhuoAvg[i][j] = sum / featureGroups.get(j).size();
    			doubleHJhuoAvg[i][j] = Math.ceil(hJhuoAvg[i][j]);
    		}
    	}
    	return doubleHJhuoAvg;
    }

    
    private static double[][] averagePooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost)
    {
    	double[][] havg = new double[shots][k]; 
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{	   			
    			//Sum calculation, gather all values of a multimodal word:
    			double sum = 0;
    			for(Integer val: featureGroups.get(j))
    			{
    				sum += featurePool[val][i];    				    			
    			}
    			//To avoid division by 0 when there are empty clusters.
    			if(featureGroups.get(j).size() > 0)
    			{
    				havg[i][j] = sum / featureGroups.get(j).size();
    			}
    			else
    			{
    				havg[i][j] = sum / 1;
    			}    			
    			
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				havg[i][j] *= boostFactor;
    			}
    		}
    	}
    	return havg;
    }
    
    private static double[][] maxPooling(ArrayList<ArrayList<Integer>> featureGroups, double featurePool[][], int shots, int k, boolean boost)
    {
    	double[][] hmax = new double[shots][k]; 
    	for(int i = 0; i < shots; i++)
    	{	
    		for(int j = 0; j < k; j++)
    		{	
    			//Select maximum value from a multimodal word ocurrence
    			double max = -1;
    			for(Integer val: featureGroups.get(j))
    			{
    				if(featurePool[val][i] > max)
    				{
    					max = featurePool[val][i];
    				}
    			}
    			hmax[i][j] = max;
    			
    			//If we want to boost co-occurrences (groups greater than 1)
    			if(featureGroups.get(j).size() > 1 && boost == true)
    			{
    				hmax[i][j] *= boostFactor;
    			}
    		}
    	}
    	return hmax;
    }    

}