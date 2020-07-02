package eu.h2020.helios_social.modules.socialgraphmining.heuristics;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;

/**
 * This class provides a {@link SocialGraphMiner} that wraps a miner predictions so as not to predict the top of a base miner's predictions
 * (e.g. DHR@k,withhold of a miner is obtained if HitRate@k is calculated over the outcome of <code>
 * new AdditionalDiscoveryMiner(miner, new RepeatAndReplyMiner(cen), withhold)</code>.
 * @author Emmanouil Krasanakis
 */
public class DifferenceMiner extends SocialGraphMiner {
	private static String parameterSplit = "@@@";
	private SocialGraphMiner discoveryMiner, baseMiner;
	private int withholdTopOfBaseMiner;

	public DifferenceMiner(SocialGraphMiner discoveryMiner, SocialGraphMiner baseMiner, int withholdTopOfBaseMiner) {
		super(discoveryMiner.getContextualEgoNetwork());
		this.discoveryMiner = discoveryMiner;
		this.baseMiner = baseMiner;
		this.withholdTopOfBaseMiner = withholdTopOfBaseMiner;
	}

	@Override
	public void newInteraction(Interaction interaction, String neighborModelParameters, InteractionType interactionType) {
		String[] params = neighborModelParameters.split(parameterSplit);
		discoveryMiner.newInteraction(interaction, params[0], interactionType);
		baseMiner.newInteraction(interaction, params[1], interactionType);
	}

	@Override
	public String getModelParameters(Interaction interaction) {
		return discoveryMiner.getModelParameters(interaction)+parameterSplit+baseMiner.getModelParameters(interaction);
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		return Utils.error("This method should not be called for differences", 0);
		//return discoveryMiner.predictNewInteraction(context, destinationNode) / (0.1+baseMiner.predictNewInteraction(context, destinationNode));
	}
	
	public HashMap<Node, Double> recommendInteractions(Context context) {
    	HashMap<Node, Double> discoveredInteractions = discoveryMiner.recommendInteractions(context);
    	HashMap<Node, Double> baseInteractions = baseMiner.recommendInteractions(context);
    	
    	List<Node> topbase = sort(baseInteractions, withholdTopOfBaseMiner);
    	List<Node> topDiscovered = sort(discoveredInteractions, withholdTopOfBaseMiner);
    	for(Node node : new ArrayList<Node>(discoveredInteractions.keySet())) 
    		if(topbase.contains(node) && topDiscovered.contains(node))
    			discoveredInteractions.remove(node);
    	/*
    	double maxBase = 0;
    	double minBase = Double.POSITIVE_INFINITY;
    	for(Node node : baseInteractions.keySet()) {
    		if(baseInteractions.get(node)<minBase)
    			minBase = baseInteractions.get(node);
    		if(baseInteractions.get(node)>maxBase)
    			maxBase = baseInteractions.get(node);
    	}
    	if(minBase!=maxBase)
	    	for(Node node : new ArrayList<Node>(discoveredInteractions.keySet()))
	    		discoveredInteractions.put(node, (baseInteractions.get(node)-minBase)/(maxBase-minBase));*/
    	
    	return discoveredInteractions;
    }

	@SuppressWarnings("rawtypes")
	protected static <T1, T2> List<T1> sort(Map<T1, T2> unsortedMap) {
		return unsortedMap
				.entrySet()
				.stream()
			    .sorted((e1, e2) -> -((Comparable) e1.getValue()).compareTo(e2.getValue()))
			    .map(e -> e.getKey())
			    .collect(Collectors.toList());
	}
	
	@SuppressWarnings("rawtypes")
	protected static <T1, T2> List<T1> sort(Map<T1, T2> unsortedMap, int topK) {
		return unsortedMap
				.entrySet()
				.stream()
			    .sorted((e1, e2) -> -((Comparable) e1.getValue()).compareTo(e2.getValue()))
			    .map(e -> e.getKey())
			    .limit(topK)
			    .collect(Collectors.toList());
	}

}