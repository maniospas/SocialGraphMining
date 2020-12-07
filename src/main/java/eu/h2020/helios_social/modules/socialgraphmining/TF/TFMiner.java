package eu.h2020.helios_social.modules.socialgraphmining.TF;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map.Entry;

import eu.h2020.helios_social.core.contextualegonetwork.Context;
import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Edge;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.GNNNodeData;
import mklab.JGNN.core.Model;
import mklab.JGNN.core.ModelBuilder;
import mklab.JGNN.core.Matrix;
import mklab.JGNN.core.Tensor;
import mklab.JGNN.core.matrix.DenseMatrix;
import mklab.JGNN.core.matrix.SparseMatrix;
import mklab.JGNN.core.matrix.SparseSymmetric;
import mklab.JGNN.core.optimizers.Adam;
import mklab.JGNN.core.tensor.DenseTensor;
import mklab.JGNN.core.util.Loss;
import mklab.JGNN.models.IdConverter;

public class TFMiner extends SocialGraphMiner {
	protected Model model;
	protected IdConverter idConverter = new IdConverter();
	protected HashMap<Edge, Double> edgeWeights = new HashMap<Edge, Double>();
	
	public TFMiner(ContextualEgoNetwork contextualEgoNetwork) {
		super(contextualEgoNetwork);
	}

	@Override
	public void newInteractionParameters(Interaction interaction, SocialGraphMinerParameters params, InteractionType interactionType) {
		if(interactionType==InteractionType.SEND)
			return;
		Context context = interaction.getEdge().getContext();
		for(Edge edge : context.getEdges()) 
			edgeWeights.put(edge, edgeWeights.getOrDefault(edge, 0.)*0.5);
		idConverter.getOrCreateId(interaction.getEdge().getSrc());
		idConverter.getOrCreateId(interaction.getEdge().getDst());
		edgeWeights.put(interaction.getEdge(), edgeWeights.get(interaction.getEdge())+1);
		
		interaction
			.getEdge()
			.getAlter()
			.getOrCreateInstance(GNNNodeData.class)
			.forceSetEmbedding((Tensor)params.get("embedding"));
		for(String edgeRepresentation : ((String)params.get("edges")).split("\\s")) {
			String[] nodes = edgeRepresentation.split("\\-");
			Edge edge = context.getOrAddEdge(	context.getContextualEgoNetwork().getOrCreateNode(nodes[0]), 
												context.getContextualEgoNetwork().getOrCreateNode(nodes[1]) );
			edgeWeights.put(edge, edgeWeights.get(interaction.getEdge()));//Math.max(edgeWeights.getOrDefault(edge, 0.), Double.parseDouble(nodes[2])));

			idConverter.getOrCreateId(context.getContextualEgoNetwork().getOrCreateNode(nodes[0]));
			idConverter.getOrCreateId(context.getContextualEgoNetwork().getOrCreateNode(nodes[1]));
		}
		
		if(context.getNodes().size()<12)
			return;
		
		Matrix W = new SparseSymmetric(idConverter.size(), idConverter.size());
		for(Edge edge : context.getEdges())
			W.put(idConverter.getId(edge.getSrc()), idConverter.getId(edge.getDst()), 1);
		W.put(	idConverter.getId(interaction.getEdge().getSrc()), 
				idConverter.getId(interaction.getEdge().getAlter()) );
		
		//System.out.println("Density:"+W.getNumNonZeroElements()/(double)W.getRows()/W.getCols());
		
		
		long dims = 10;
		Matrix H0 = new DenseMatrix(idConverter.size(), dims);
		for(Node node : context.getNodes()) {
			long i = idConverter.getId(node);
			Tensor previousEmbedding = node.getOrCreateInstance(GNNNodeData.class).getEmbedding();
			for(long dim=0;dim<dims;dim++)
				H0.put(i, dim, previousEmbedding.get(dim));
		}
		model = new ModelBuilder()
				.var("u")
				.var("v")
				.constant("W", W)
				.param("H0", H0)
				.param("DistMult", new DenseTensor(dims).setToRandom().setToNormalized())
				.param("W1", new DenseMatrix(dims, dims).setToRandom().setToNormalized())
				.param("W2", new DenseMatrix(dims, dims).setToRandom().setToNormalized())
				.operation("H1 = W * H0 * W1 + H0 * W2")
				.operation("sim = sigmoid( sum(H1[u].H1[v].DistMult) )")
				.out("sim")
				.assertBackwardValidity()
				.getModel();
		
		long numEdges = W.getNumNonZeroElements()*4;
		Tensor weights = new DenseTensor(numEdges);
		Tensor labels = new DenseTensor(numEdges);
		Tensor uList = new DenseTensor(numEdges);
		Tensor vList = new DenseTensor(numEdges);
		long pos = 0;
		for(Edge edge : context.getEdges()) {
			long u = (long)idConverter.getId(edge.getSrc());
			long v = (long)idConverter.getId(edge.getDst());
			
			uList.put(pos, u);
			vList.put(pos, v);
			labels.put(pos, 1);
			weights.put(pos, edgeWeights.get(edge));
			pos += 1;

			long neg1 = (int)(W.getRows()*Math.random());
			long neg2 = (int)(W.getRows()*Math.random());
			while(W.get(neg1,neg2)!=0 || neg1==neg2) {
				neg1 = (int)(W.getRows()*Math.random());
				neg2 = (int)(W.getRows()*Math.random());
			}
			uList.put(pos, neg1);
			vList.put(pos, neg2);
			weights.put(pos, 1);
			pos += 1;
		}
		for(int epoch=0;epoch<10;epoch++)
			model.trainSample(new Adam(1), Arrays.asList(uList, vList),  Arrays.asList(labels), Arrays.asList(weights));
		
		for(Node node : context.getNodes()) {
			//if(node!=context.getContextualEgoNetwork().getEgo())
			//	continue;
			long i = idConverter.getId(node);
			Tensor previousEmbedding = node.getOrCreateInstance(GNNNodeData.class).getEmbedding();
			for(long dim=0;dim<dims;dim++)
				previousEmbedding.put(dim, H0.get(i, dim));
		}
	}

	protected static <ObjectType> ObjectType sampleFrom(HashMap<ObjectType, Double> probs) {
		double sum = 0;
		for(double prob : probs.values())
			sum += prob;
		if(sum==0)
			throw new RuntimeException("Make sure that sampling weights are provided for at least one element");
		double pos = sum*Math.random();
		for(ObjectType obj : probs.keySet()) {
			pos -= probs.get(obj);
			if(pos<=0)
				return obj;
		}
		throw new RuntimeException("Failed to sample (should not occur)");
	}

	@Override
	public SocialGraphMinerParameters getModelParameterObject(Interaction interaction) {
		Context context = interaction.getEdge().getContext();
		SocialGraphMinerParameters params = new SocialGraphMinerParameters();
		params.put("embedding", context
									.getContextualEgoNetwork()
									.getEgo()
									.getOrCreateInstance(GNNNodeData.class)
									.getEmbedding());
		String edges = "";
		for(Edge edge : context.getEdges())
			edges += " "+edge.getSrc().getId()+"-"+edge.getDst().getId()+"-"+edgeWeights.get(edge);
		params.put("edges", edges.trim());
		return params;
	}

	@Override
	public double predictNewInteraction(Context context, Node destinationNode) {
		if(model==null)
			return 0;
		int u = idConverter.getId(context.getContextualEgoNetwork().getEgo());
		int v = idConverter.getId(destinationNode);
		return model.predict(Arrays.asList(Tensor.fromDouble(u), Tensor.fromDouble(v))).get(0).get(0);
	}

}
