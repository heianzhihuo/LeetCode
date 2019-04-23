import java.util.Scanner;


public class ByteDance3 {

	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {// 注意，如果输入是多个测试用例，请通过while循环处理多个测试用例
        	N = in.nextInt();
        	map = new int[N][N];
        	visited = new boolean[N];
        	minc = Integer.MAX_VALUE;
            int i,j;
            for(i=0;i<N;i++)
            	for(j=0;j<N;j++)
            		map[i][j]= in.nextInt();
        	tryNext(0,0,N);
        	System.out.println(minc);
        }
	}
	
	static int minc = Integer.MAX_VALUE;
	static int N;
	static boolean visited[];
	static int map[][];
	
	public static void tryNext(int x,int cur,int left) {
		if(visited[x])
			return;
		left--;
		if(left==0) {
			if(cur+map[x][0]<minc)
				minc = cur+map[x][0];
			return;
		}
		if(cur+map[x][0]>=minc)
			return;
		visited[x] = true;
		for(int i=0;i<N;i++) 
			if(i!=x && !visited[i])
				tryNext(i, cur+map[x][i], left);
		visited[x] = false; 
	}
	

}
