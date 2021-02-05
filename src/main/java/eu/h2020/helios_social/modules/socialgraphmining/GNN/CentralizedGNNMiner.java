package eu.h2020.helios_social.modules.socialgraphmining.GNN;

import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.storage.NoStorage;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;

/**
 * This class provides an implementation of a {@link SocialGraphMiner} based on
 * a Graph Neural Network (GNN) architecture. Its parameters can be adjusted using a number of
 * setter methods.
 * 
 * @author Emmanouil Krasanakis
 */
public class CentralizedGNNMiner extends SocialGraphMiner {
	private static ContextualEgoNetwork contextualEgoNetwork = ContextualEgoNetwork.createOrLoad(new NoStorage(""), "test", null);
	private static GNNMiner gnnMiner = new GNNMiner(contextualEgoNetwork);
	private static Node lastNode = null;
	
	public CentralizedGNNMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}
	public GNNMiner getBaseMiner() {
		return gnnMiner;
	}
	public void setReferenceNode(Node node) {
		lastNode = contextualEgoNetwork.getOrCreateNode(node.getId(), null);
	}
	@Override
	public void newInteractionParameters(Interaction interaction, SocialGraphMinerParameters neighborModelParameters, InteractionType interactionType) {
		if(interactionType!=InteractionType.SEND)
			return;
		gnnMiner.newInteractionParameters(contextualEgoNetwork.getOrCreateContext("test_context")
				.addEdge(
					contextualEgoNetwork.getOrCreateNode(interaction.getEdge().getSrc().getId(), null),
					contextualEgoNetwork.getOrCreateNode(interaction.getEdge().getDst().getId(), null))
				.addDetectedInteraction(null)
				, null, interactionType);
		
	}
	@Override
	public SocialGraphMinerParameters getModelParameterObject(Interaction interaction) {
		return null;
	}
	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		return gnnMiner.predictNewInteraction(contextualEgoNetwork.getOrCreateContext("test_context"), 
				contextualEgoNetwork.getOrCreateNode(lastNode.getId(), null),
				contextualEgoNetwork.getOrCreateNode(destinationNode.getId(), null));
	}
	@Override
    public HashMap<Node, Double> recommendInteractions(Context context) {
    	HashMap<Node, Double> scores = new HashMap<Node, Double>();
    	if(lastNode!=null) {
	    	Node ego = contextualEgoNetwork.getOrCreateNode(lastNode.getId(), null);
	    	for(Node node : context.getNodes()) {
	    		node = contextualEgoNetwork.getOrCreateNode(node.getId(), null);
	    		if(node!=ego && (context.getEdge(ego, node)!=null || context.getEdge(node, ego)!=null))
	    			scores.put(node, predictNewInteraction(context, node));
	    	}
    	}
    	return scores;
    }
}
