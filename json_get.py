import subprocess
import glob

JSON_FILENAME = "output.json"

#https://community.cloudera.com/t5/CDH-Manual-Installation/amp-quot-RuntimeException-core-site-xml-not-found-amp-quot/td-p/87590
# So, I assume that the problem comes from the code where HIVE_CONF_DIR is appended to HADOOP_CONF_DIR. 
#
os.environ['HADOOP_CONF_DIR'] = "/opt/cloudera/parcels/CDH-6.1.0-1.cdh6.1.0.p0.770702/lib/spark/conf/yarn-conf"
#os.environ['HADOOP_CONF_DIR'] = "/etc/spark/conf/yarn-conf"

#args = ['hdfs', 'dfs', '-ls', '/tmp/']
HDFS_DIR = '/tmp/'
LOCAL_DIR = '.'

args = ['hdfs', 'dfs', '-get', HDFS_DIR + JSON_FILENAME, LOCAL_DIR]

try:
  proc = subprocess.run(args,stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  
except:
  import traceback
  traceback.print_exc()
  print("Error.")