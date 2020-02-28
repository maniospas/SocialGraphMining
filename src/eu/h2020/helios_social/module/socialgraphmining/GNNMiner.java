package eu.h2020.helios_social.module.socialgraphmining;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Edge;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.module.socialgraphmining.GNN.Tensor;

/**
 * This class provides an implementation of a {@link SocialGraphMiner} based on
 * a Graph Neural Network (GNN) architecture.
 * 
 * @author Emmanouil Krasanakis
 */
public class GNNMiner extends SocialGraphMiner {
	private ContextualEgoNetwork contextualEgoNetwork;
	private Tensor relation;
	private Tensor egoId;
	private HashMap<Node, Tensor> alterIds;
	private HashMap<Node, Tensor> alterNegativeIds;
	private static int dims = 10;
	private ArrayList<Node> prevAlters = new ArrayList<Node>();
	private HashMap<Interaction, Integer> interactionOrder = new HashMap<Interaction, Integer>();
	
	/**
	 * Constructor of the GNN model
	 * @param contextualEgoNetwork The contextual ego network library instance this model works on
	 */
    public GNNMiner(ContextualEgoNetwork contextualEgoNetwork) {
    	super(contextualEgoNetwork);
    	this.contextualEgoNetwork = contextualEgoNetwork;
    	egoId = new Tensor(dims);
    	egoId.setRandom();
    	egoId.setNormalize();
    	if(relation==null) {
    		relation = new Tensor(dims);
    		relation.setOnes();
    		relation.setNormalize();
    	}
    	alterIds = new HashMap<Node, Tensor>();
    	alterNegativeIds = new HashMap<Node, Tensor>();
    }

	private double getVotingStrength(Edge edge, Interaction current) {
		if(edge.getInteractions().size()==0)
			return 0;
		if(current!=null && edge.getAlter()==current.getEdge().getAlter())
			return 1;
		return 0;
	}

	@Override
	public void newInteraction(Interaction interaction) {
	}
	
	public Tensor derivative(Tensor otherAlterId, double target, double weight) {
		double output = 1./(1+Math.exp(-relation.dot(egoId, otherAlterId)));
		double partial = (1-target)*output - target*(1-output);
		partial *= weight;
		return relation.multiply(otherAlterId).multiply(partial);
	}

	@Override
	public synchronized void newInteraction(Interaction interaction, String neighborModelParameters, boolean isReply) {
		if(!interactionOrder.containsKey(interaction))
			interactionOrder.put(interaction, interactionOrder.size());
		if(!isReply) {
			prevAlters.add(interaction.getEdge().getAlter());
			if(prevAlters.size()>20)
				prevAlters.remove(0);
		}
		String[] params = neighborModelParameters.split("\\|");
		Tensor alterId = new Tensor(params[0]);
		//Tensor alterRelation = new Tensor(params[1]);
		alterNegativeIds.put(interaction.getEdge().getAlter(), new Tensor(params[2]));
		alterIds.put(interaction.getEdge().getAlter(), alterId);
		int count = 0;
		while(true) {
			Tensor accum = egoId.zeroCopy();
			for(Edge edge : contextualEgoNetwork.getCurrentContext().getEdges()) {
				Tensor otherAlterId = alterIds.get(edge.getAlter());
				if(otherAlterId==null || edge.getSrc()!=contextualEgoNetwork.getEgo() || edge.getInteractions().size()<=0)
					continue;
				double weight = getVotingStrength(edge, interaction);
				if(weight==0)
					continue;
				accum = accum.add(derivative(otherAlterId, 1, weight));
				otherAlterId = alterNegativeIds.get(edge.getAlter());
				accum = accum.add(derivative(otherAlterId, 1, 0.1*weight));
			}
			//accum.add(egoId.multiply(0.1));
		//	if(count>1)
			//	lastSimilarity = egoId.dot(accum.multiply(-1))/accum.norm();
			Tensor prevEgo = egoId;
			egoId = egoId.add(accum.multiply(-1)).normalized();
			if(prevEgo.subtract(egoId).norm()<0.001)
				break;
			count++;
		}
		//relation = relation.add(relationAccum.multiply(-0.01));
		//System.out.println(relation);
		//double sim = 1;//egoId.dot(alterId)/egoId.norm()/alterId.norm();
		//relation = relation.multiply(1-sim*0.5).add(alterRelation.multiply(sim*0.5));
	}

	@Override
	public String getModelParameters(Interaction interaction) {
		Tensor negative = egoId.zeroCopy();
		double count = 0;
		for(Edge edge : contextualEgoNetwork.getCurrentContext().getEdges()) {
			double weight = 1;
			if(edge.getSrc()!=interaction.getEdge().getAlter() && edge.getSrc()!=contextualEgoNetwork.getEgo() && weight!=0 && alterIds.containsKey(edge.getAlter())) {
				negative = negative.add(alterIds.get(edge.getAlter()).multiply(weight));
				count += weight;
			}
		}
		if(count!=0) 
			negative = negative.multiply(1./count);
		//negative.normalize();
		return egoId.toString()+"|"+relation.toString()+"|"+negative.toString();
	}

	@Override
	public Map<Edge, Double> predictOutgoingInteractions() {
		HashMap<Edge, Double> evaluations = new HashMap<Edge, Double>();
		for(Edge edge : contextualEgoNetwork.getCurrentContext().getEdges()) {
			if(edge.getSrc()!=contextualEgoNetwork.getEgo() || edge.getInteractions().isEmpty())
				continue;
			Tensor alterId = alterIds.get(edge.getAlter());
			if(alterId!=null)
				evaluations.put(edge, relation.dot(alterId, egoId));
		}
		return evaluations;
	}

	@Override
	public Map<String, Double> evaluate(Interaction interaction) {
		Map<String, Double> result =  new LinkedHashMap<String, Double>();

		//result.put("New edge", interaction.getEdge().getInteractions().size()<=1?1.:0);
		if(interaction.getEdge().getInteractions().size()<=1) {//>1 keeps only the next friendship predictions, <=1 keeps the prediction of next interaction among friends
			return result;
		}
		if(!alterIds.containsKey(interaction.getEdge().getAlter()) || interaction.getEdge().getSrc()!=contextualEgoNetwork.getEgo())
			return result;
		Map<Edge, Double> evaluations = predictOutgoingInteractions();
		if(evaluations.size()<=1)
			return result;
		double topk = topK(evaluations, interaction.getEdge());
		result.put("HR@5", topk<=5?1.:0.);
		result.put("HR@3", topk<=3?1.:0.);
		result.put("HR@1", topk<=1?1.:0.);

		evaluations = new HashMap<Edge, Double>();
		for(Edge edge : contextualEgoNetwork.getCurrentContext().getEdges()) 
			if(alterIds.get(edge.getAlter())!=null)
				evaluations.put(edge, Math.random());
		topk = topK(evaluations, interaction.getEdge());
		result.put("Random HR@1", topk<=1?1.:0.);

		if(prevAlters.size()>0) {
			result.put("Last interactions HR@5", posOfLast(prevAlters)<=5?1.:0.);
			result.put("Last interactions HR@3", posOfLast(prevAlters)<=3?1.:0.);
			result.put("Last interactions HR@1", posOfLast(prevAlters)<=1?1.:0.);
		}
		
		return result;
	}
	
	protected static int posOfLast(ArrayList<Node> nodes) {
		int n = nodes.size();
		for(int i=1;i<n-1;i++)
			if(nodes.get(n-1-i)==nodes.get(n-1))
				return i;
		return Integer.MAX_VALUE;
	}
	
	protected static int topK(Map<Edge, Double> evaluations, Edge target) {
		double assignedEvaluation = evaluations.get(target);
		int topk = 0;
		for(double value : evaluations.values())
			if(value>=assignedEvaluation)
				topk += 1;
		return topk;
	}
}
