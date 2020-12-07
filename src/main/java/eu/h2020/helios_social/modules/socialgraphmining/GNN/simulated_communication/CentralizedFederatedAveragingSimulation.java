package eu.h2020.helios_social.modules.socialgraphmining.GNN.simulated_communication;

import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Node;
import mklab.JGNN.core.Tensor;

public class CentralizedFederatedAveragingSimulation extends EmbeddingExchangeProtocol{
	private static double currentTime = 0;
	private static boolean increaseTimeNextRegister = true;
	private static HashMap<String, HashMap<String, Tensor>> estimations = new HashMap<String, HashMap<String, Tensor>>();
	private static HashMap<String, HashMap<String, Double>> times = new HashMap<String, HashMap<String, Double>>();
	private static HashMap<String, Double> lastTimes = new HashMap<String, Double>();
	private double availability;

	public CentralizedFederatedAveragingSimulation(double availability) {
		this.availability = availability;
	}
	
	@Override
	public Tensor requestEmbeddings(Node ego, Node alter) {
		String alterId = alter.getId();
		if(!estimations.containsKey(alterId))
			return null;
		increaseTimeNextRegister = true;
		Tensor sum = null;
		double totalWeight = 0;
		for(String estimatorId : estimations.get(alterId).keySet()) {
			if(Math.random()>availability)
				continue;
			if(sum==null)
				sum = estimations.get(alterId).get(estimatorId).zeroCopy();
			double weight = Math.exp(-0.001*(times.get(alterId).get(estimatorId)-lastTimes.get(alterId)));
			sum.selfAdd(estimations.get(alterId).get(estimatorId).multiply(weight));
			totalWeight += weight;
		}
		if(sum==null)
			return null;
		return sum.selfMultiply(1./totalWeight);
	}

	@Override
	public void registerEmbeddings(Node ego, Node alter, Tensor parameters) {
		if(increaseTimeNextRegister) 
			currentTime += 1;
		increaseTimeNextRegister = false;
		String egoId = ego.getId();
		String alterId = alter.getId();
		if(!estimations.containsKey(alterId)) {
			estimations.put(alterId, new HashMap<String, Tensor>());
			times.put(alterId, new HashMap<String, Double>());
		}
		estimations.get(alterId).put(egoId, parameters.copy());
		times.get(alterId).put(egoId, currentTime);
		lastTimes.put(alterId, currentTime);
	}

}
