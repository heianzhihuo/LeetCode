import java.util.*;



public class LeetCode5 {

	public int findLongestChain(int[][] pairs) {
        int n = pairs.length;
        if(n<=1)
            return n;
        Arrays.sort(pairs,(a,b)->a[0]-b[0]);
        int ret = 1;
        int end = pairs[0][1];
        for(int i=1;i<n;i++)
            if(pairs[i][0]>end){
                ret++;
                end = pairs[i][1];
            }else end = Math.min(end,pairs[i][1]);
        return ret;
    }
	
	public int lenLongestFibSubseq(int[] A) {
        int n = A.length;
        if(n<=2)
            return n;
        int ret = 2;
        HashSet<Integer> pre[] = new HashSet[n];
        int dp[] = new int[n];
        dp[0] = 1;
        pre[0] = new HashSet<>();
        dp[1] = 2;
        pre[1] = new HashSet<>();
        pre[1].add(A[0]);
        for(int i=2;i<n;i++){
            dp[i] = 1;
            for(int j=0;j<i;j++)
                if(dp[j]+1>dp[i]){
                    if(dp[j]==1 || pre[j].contains(A[i]-A[j])){
                        dp[i] = dp[j]+1;
                        pre[i] = new HashSet<>();
                        pre[i].add(A[j]);
                    }
                }else if(dp[j]+1==dp[i]){
                    if(dp[j]==1 || pre[j].contains(A[i]-A[j])){
                        pre[i].add(A[j]);
                    }
                }
            ret = Math.max(ret,dp[i]);
        }
        return ret;
    }
	
	public static void main(String[] args) {
        LeetCode5 test = new LeetCode5();
        
        List<Integer> integers = Arrays.asList(1,2,13,4,15,6,17,8,19);
        IntSummaryStatistics stats = integers.stream().mapToInt((x) ->x).summaryStatistics();
        System.out.println("列表中最大的数 : " + stats.getMax());
        
        
        
        
        
        
        
        
        int AA[] = {2,4,7,8,9,10,14,15,18,23,32,50};
		System.out.println(test.lenLongestFibSubseq(AA));
		
		
		
		
		
		
		
		
		Scanner in = new Scanner(System.in);
        while (in.hasNextInt()) {//注意while处理多个case
            int m = in.nextInt();
            int n = in.nextInt();
            int A[] = new int[n];
            int i,j,k,count=0;
            for(i=0;i<n;i++)
                A[i] = in.nextInt();
            Arrays.sort(A);
            if(A[0]!=1){
                System.out.println(-1);
                continue;
            }
            int min[] = new int[m+1];
            int max;
            min[1] = 1;
            for(i=1;i<=m;i++) {
            	max = i;
            	for(j=0;j<n;j++) {
            		if(i-A[j]>=1 && i-A[j]<i) {
            			if(min[i-A[j]]+1<max)
            				max = min[i-A[j]]+1;
            		}else break;
            	}
            	min[i] = max;
            }
            System.out.println(min[m]);
        }
    }
	

}
