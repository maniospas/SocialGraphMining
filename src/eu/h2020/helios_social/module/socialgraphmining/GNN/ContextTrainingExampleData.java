package eu.h2020.helios_social.module.socialgraphmining.GNN;

import java.util.ArrayList;

/**
 * This class provides a storage structure that organizes a list of {@link TrainingExample} data to be stored in the 
 * contextual ego network's contexts.
 * 
 * It is indented to be used as a dynamically created instance on contexts (which are cross module components)
 * by calling <code>context.getOrCreateInstance(GNNNodeData.class)</code> to either retrieve of create it.
 * 
 * @author Emmanouil Krasanakis
 */
public class ContextTrainingExampleData {
	private static double removalThreshold = 0.1;
	private ArrayList<TrainingExample> trainingExamples = null;
	
	public ContextTrainingExampleData() {}
	
	/**
	 * Grants direct access to a list of training examples to traverse of edit.
	 * @return An array list of training examples.
	 */
	public synchronized ArrayList<TrainingExample> getTrainingExampleList() {
		if(trainingExamples==null)
			trainingExamples = new ArrayList<TrainingExample>();
		return trainingExamples;
	}
	
	/**
	 * Calls the {@link TrainingExample#degrade(double)} operation for each {@link TrainingExample} in the
	 * data (e.g. to reduce all weights) and removes those that end up on a very small weight.
	 * @param factor The degrading factor.
	 */
	public synchronized void degrade(double factor) {
		for(TrainingExample trainingExample : new ArrayList<TrainingExample>(getTrainingExampleList())) {
			trainingExample.degrade(factor);
			if(trainingExample.getWeight() < removalThreshold)
				trainingExamples.remove(trainingExample);
		}
	}
}
