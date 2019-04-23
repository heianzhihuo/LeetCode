import java.util.HashMap;
import java.util.Scanner;

/**
 * @author WenWei
 * @date 2019年4月14日
 * @time 上午9:45:07
 */
public class ByteDance2 {

	public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int N = in.nextInt();
        for(;N>=0;N--) {
        	int M = in.nextInt();
        	in.nextLine();
        	if(M<=1)
        		System.out.println(M);
        	else {
        		int maxLength = 0;
        		int curfe[] = new int[0];
        		int cur[] = new int[0];
        		int i,j,k;
        		for(;M>0;M--) {
        			int s = in.nextInt();
        			int nextfe[] = new int[s*2];
        			for(k=0;k<s*2;k++)
        				nextfe[k] = in.nextInt(); 
        			int next[] = new int[s];
        			for(i=0;i<s;i++) {
        				boolean flag = false;
        				for(j=0;j<cur.length;j++)
        					if(nextfe[i*2]==curfe[j*2] && nextfe[i*2+1]==curfe[j*2+1]) {
        						flag = true;
        						break;
        					}
        				if(flag==true)
        					next[i] = cur[j]+1;
        				if(next[i]>maxLength)
        					maxLength = next[i];
        				
        			}
        			cur = next;
        			curfe = nextfe;
        		}
        		System.out.println(maxLength+1);
        	}
        }
	}
}
