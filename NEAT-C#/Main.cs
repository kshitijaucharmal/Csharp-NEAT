using System;
using NEAT;
using System.Collections.Generic;

public class main{
	public static void Main(){
		ConnectionHistory ch = new ConnectionHistory(4, 2);
		Genome g = new Genome(ch, true);
		main m = new main();

		m.Show(g, ch);
		g.AddInputs(new float[]{0.3f, 0.4f, 0.5f, 0.7f});

		Console.WriteLine(g.GiveOutputs().Length);
	}

	private void Show(Genome g, ConnectionHistory ch){
		Genome.showGenome(g.Connections, "G1");
		Console.WriteLine();
		Genome.showGenome(ch.AllConnections, "All Connections");
		Console.WriteLine();
	}
}

// ---------------------------------------------- Config ----------------------------------------------
namespace NEAT{
	public class Config{
		public static int input_layer = 0;
		public static int output_layer = 10;
	}
}

// --------------------------------------------- Node class---------------------------------------------
namespace NEAT{
	public class Node{
		public int number, layer;
		public List<Connection> InConnections = new List<Connection>();

		public float sum = 0f;
		public float outputValue = 0f;

		public Node(int n, int l){
			number = n;
			layer = l;
		}

		public void Calculate(){
			if(layer == Config.input_layer){
				Console.WriteLine("This is the input layer");
				return;
			}

			foreach(Connection c in InConnections){
				sum += c.weight * c.in_node.outputValue;
			}

			outputValue = Activate(sum);
		}

		public float Activate(float x){
			return 1f / (1f + (float)Math.Exp(-x));
		}

		public Node copy(){
			Node n = new Node(number, layer);
			n.outputValue = outputValue;
			n.sum = sum;
			return n;
		}
	}
}

// --------------------------------------------- Connection class---------------------------------------------
namespace NEAT{
	public class Connection{
		public Node in_node, out_node;
		public float weight;
		public bool enabled;
		public int inno;
		protected Random rand;
		public Connection(Node i, Node o){
			rand = new Random();
			in_node = i;
			out_node = o;

			weight = (float)rand.NextDouble() * 4 - 2;
			enabled = true;
			inno = -1;
		}

		public void showConn(){
			Console.WriteLine(inno + ". " + in_node.number + '(' + in_node.layer + ')' + " -> " + out_node.number + '(' + out_node.layer + ')' + " " + weight + " " + enabled);
		}

		// Mutating funtions
		public void Randomize(){
			weight = (float)rand.NextDouble() * 4 - 2;
		}

		public void Toggle(){
			enabled = !enabled;
		}

		public Connection copy(){
			Connection c = new Connection(in_node.copy(), out_node.copy());
			c.weight = weight;
			c.inno = inno;
			c.enabled = enabled;
			return c;
		}
	}
}

// --------------------------------------------- ConnectionHistory class---------------------------------------------
namespace NEAT{
	public class ConnectionHistory{
		public int inputs, outputs;
		public List<Connection> AllConnections = new List<Connection>();
		public int global_inno = 0;

		public ConnectionHistory(int i, int o){
			inputs = i;
			outputs = o;
		}

		public Connection Exists(Node n1, Node n2){
			foreach(Connection c in AllConnections){
				if(c.in_node.number == n1.number && c.out_node.number == n2.number){
					return c;
				}
			}
			return null;
		}
	}
}

// --------------------------------------------- Genome class---------------------------------------------
namespace NEAT{
	public class Genome{
		public int inputs, outputs;
		public ConnectionHistory ch;

		public List<Node> Nodes = new List<Node>();
		public List<Connection> Connections = new List<Connection>();

		public int total_nodes = 0;
		Random rand = new Random();

		public Genome(ConnectionHistory ch, bool create){
			this.ch = ch;
			inputs = ch.inputs;
			outputs = ch.outputs;

			if(create){
				CreateNetwork();
			}
		}

		public void CreateNetwork(){
			for(int i = 0; i < inputs; i++) Nodes.Add(new Node(total_nodes++, Config.input_layer));
			for(int i = 0; i < outputs; i++) Nodes.Add(new Node(total_nodes++, Config.output_layer));

			for(int i = 0; i < inputs * outputs; i++) if(rand.NextDouble() < 0.6) AddConnection();
		}

		public void AddConnection(){
			Node n1 = Nodes[rand.Next(Nodes.Count)];
			Node n2 = Nodes[rand.Next(Nodes.Count)];

			while(n1.layer == Config.output_layer) n1 = Nodes[rand.Next(Nodes.Count)];
			while(n2.layer == Config.input_layer || n2.layer <= n1.layer) n2 = Nodes[rand.Next(Nodes.Count)];

			Connection c = ch.Exists(n1, n2);
			Connection x = new Connection(n1, n2);

			if(c != null){
				x.inno = c.inno;
				if(!Exists(x.inno)){
					Connections.Add(x);
					n2.InConnections.Add(x);
				}
			}
			else{
				x.inno = ch.global_inno;
				ch.global_inno++;
				Connections.Add(x);
				ch.AllConnections.Add(x.copy());
				n2.InConnections.Add(x);
			}
		}

		public void AddInputs(float[] ins){
			if(ins.Length != inputs){
				Console.WriteLine("Wrong Length Inputs");
				return;
			}

			for(int i = 0; i < inputs; i++){
				Nodes[i].outputValue = ins[i];
			}
		}

		public float[] GiveOutputs(){
			List<float> outs = new List<float>();
			for(int j = Config.input_layer+1; j < Config.output_layer+1; j++){
				for(int i = 0; i < Nodes.Count; i++){
					if(Nodes[i].layer == j){
						Nodes[i].Calculate();
					}

					if(Nodes[i].layer == Config.output_layer){
						outs.Add(Nodes[i].outputValue);
					}
				}
			}

			return outs.ToArray();
		}

		public bool Exists(int nn){
			foreach(Connection c in Connections){
				if(c.inno == nn) return true;
			}
			return false;
		}

		public void AddNode(){
			Nodes.Add(new Node(total_nodes++, rand.Next(Config.input_layer+1, Config.output_layer)));
		}

		public static void showGenome(List<Connection> g, string title){
			Console.WriteLine("---------------- " + title + " -----------------------");
			foreach(Connection c in g){
				c.showConn();
			}
		}
	}
}