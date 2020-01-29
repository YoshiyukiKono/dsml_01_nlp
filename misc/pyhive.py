from pyhive import hive
from TCLIService.ttypes import TOperationState
cursor = hive.connect('ip-10-0-0-55.ap-northeast-1.compute.internal', port=10000).cursor()
cursor.execute('show tables', async=True)

status = cursor.poll().operationState
while status in (TOperationState.INITIALIZED_STATE, TOperationState.RUNNING_STATE):
    logs = cursor.fetch_logs()
    for message in logs:
        print(message)
    status = cursor.poll().operationState

print(cursor.fetchall())
