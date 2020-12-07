package eu.h2020.helios_social.modules.socialgraphmining.GNN.simulated_communication;

import eu.h2020.helios_social.core.contextualegonetwork.Node;
import mklab.JGNN.core.Tensor;

public abstract class EmbeddingExchangeProtocol {
	public EmbeddingExchangeProtocol() {}
	public abstract Tensor requestEmbeddings(Node ego, Node alter);
	public abstract void registerEmbeddings(Node ego, Node alter, Tensor parameters);
}
