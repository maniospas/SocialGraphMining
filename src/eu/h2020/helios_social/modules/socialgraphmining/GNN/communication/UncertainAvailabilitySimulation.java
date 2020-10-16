package eu.h2020.helios_social.modules.socialgraphmining.GNN.communication;

import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Tensor;

public class UncertainAvailabilitySimulation extends EmbeddingExchangeProtocol {
	private static HashMap<String, Tensor> lastEmbeddings = new HashMap<String, Tensor>();
	private double availability;
	
	public UncertainAvailabilitySimulation(double availability) {
		this.availability = availability;
	}

	@Override
	public Tensor requestEmbeddings(Node ego, Node alter) {
		if(Math.random()>availability)
			return null;
		return lastEmbeddings.get(alter.getId());
	}

	@Override
	public void registerEmbeddings(Node ego, Node alter, Tensor parameters) {
		if(ego == alter)
			lastEmbeddings.put(ego.getId(), parameters.copy());
	}

}
