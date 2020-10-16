package eu.h2020.helios_social.modules.socialgraphmining.GNN.communication;

import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Tensor;

public abstract class EmbeddingExchangeProtocol {
	public EmbeddingExchangeProtocol() {}
	public abstract Tensor requestEmbeddings(Node ego, Node alter);
	public abstract void registerEmbeddings(Node ego, Node alter, Tensor parameters);
}
