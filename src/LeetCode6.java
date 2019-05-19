import java.util.*;
import java.util.Scanner;

public class LeetCode6 {
	
	private static int totalPrice(int categoryCount, int totalVolume, int totalWeight, int[] volume, int[] weight,
			int[] stock, int[] price, int[] itemType) {

		return 0;
	}
	/*210. Course Schedule II
	 * 课程排列问题，返回一个可行的拓扑序列*/
	public int[] findOrder(int numCourses, int[][] prerequisites) {
        int n = numCourses;
        int i;
        adj = new List[n];
        for(i=0;i<n;i++)
            adj[i] = new LinkedList<>();
        for(i=0;i<prerequisites.length;i++)
            adj[prerequisites[i][1]].add(prerequisites[i][0]);
        visited = new boolean[n];
        finished = new boolean[n];
        ret = new int[n];
        index = n-1;
        for(i=0;i<n;i++)
            if(DFS(i))
                return new int[0];
        return ret;
    }
    int []ret;
    int index;
    List<Integer>[] adj;
    boolean []visited;
    boolean []finished;
    boolean DFS(int v){
        if(finished[v])
            return false;
        else if(visited[v])
            return true;
        visited[v] = true;
        for(int c:adj[v])
            if(DFS(c))
                return true;
        finished[v]= true; 
        ret[index] = v;
        index--;
        return false;
    }
	
    /**/
    int heap[];
    int n;
    int pop(){
        n--;
        int v = heap[n];
        heap[n] = heap[0];
        int i = 0;
        int j = (i+1)*2-1;
        while(j<n){
            if(j+1<n && heap[j+1]>heap[j])
                j++;
            if(heap[j]<=v)
                break;
            heap[i] = heap[j];
            i = j;
            j = (i+1)*2-1;
        }
        heap[i] = v;
        return heap[n];
    }
    void insert(int v){
        //heap[n] = v;
        int i = n;
        while(i>0){
            int j = (i-1)/2;
            if(heap[j]>=v)
                break;
            heap[i] = heap[j];
            i = j;
        }
        heap[i] = v;
        n++;
    }
    
    public int lastStoneWeight(int[] stones) {
        heap = new int[stones.length];
        for(int x:stones)
           insert(x);
        //System.out.println(heap[0]);
        while(n>1){
            int x = pop();
            int y = pop();
            if(x!=y)
                insert(x-y);
        }
        if(n==0)
            return 0;
        else
            return heap[0];
    }
    
    
	public static void main(String[] args) {
		LeetCode6 test = new LeetCode6();
		int[][] prerequisites = {{0,1}};
		int []ress = test.findOrder(2, prerequisites);
		for(int c:ress)
			System.out.print(c+",");
		
		
		
		
		
		
		Scanner in = new Scanner(System.in);
		String[] line = in.nextLine().split(",");
//总共商品种类
		int categoryCount = Integer.valueOf(line[0]);
//快递体积
		int totalVolume = Integer.valueOf(line[1]);
//快递重量
		int totalWeight = Integer.valueOf(line[2]);

//物品体积
		int[] volume = new int[50];
//重量
		int[] weight = new int[50];
//件数
		int[] stock = new int[50];
//价格
		int[] price = new int[50];
//类型
		int[] itemType = new int[50];

		for (int i = 1; i <= categoryCount; i++) {
			line = in.nextLine().split(",");
			volume[i] = Integer.valueOf(line[0]);
			weight[i] = Integer.valueOf(line[1]);
			stock[i] = Integer.valueOf(line[2]);
			price[i] = Integer.valueOf(line[3]);
			itemType[i] = Integer.valueOf(line[4]);
		}

		in.close();

		System.out.println(totalPrice(categoryCount, totalVolume, totalWeight, volume, weight, stock, price, itemType));

	}

}
