__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import os
import paramiko

# -------------------------------------------------------------------------------------------------------------------- #
# Livestock SSH Functions


def check_for_remote_folder(sftp_connect: paramiko.SSHClient.open_sftp,
                            folder_to_check: str, check_for: str) -> bool:
    """
    Checks if remote folder exists in the desired location.
    If do exists the function returns True.
    Otherwise is creates the folder and then returns True.

    :param sftp_connect: SFTP connection
    :type sftp_connect: paramiko.SSHClient().open_sftp()
    :param folder_to_check: Path where there should be looked.
    :type folder_to_check: str
    :param check_for: Folder, which existence is wanted.
    :type check_for: str
    :return: True on success
    :rtype: bool
    """

    dir_contains = sftp_connect.listdir(folder_to_check)

    found = False

    for name in dir_contains:
        if name == check_for:
            found = True
            break
        else:
            pass

    if found:
        return True
    else:
        sftp_connect.mkdir(folder_to_check + '/' + check_for)
        return True


def ssh_connection():
    """
    This function opens up a SSH connection to a remote machine (Ubuntu-machine)
     based on inputs from the in_data.txt
    file. Once it is logged in then function activates the anaconda environment
    livestock_env, sends the commands,
    awaits their completion (by looking for a out.txt file, which is only
    written upon completion of the commands)
    and returns the wanted files back to the local machine.
    """

    # Open input text file
    local_folder = r'C:\livestock\ssh'
    in_data = '\\in_data.txt'

    file_obj = open(local_folder + in_data, 'r')
    data = file_obj.readlines()
    file_obj.close()

    # Get data
    ip = data[0][:-1]
    port = int(data[1][:-1])
    user = data[2][:-1]
    pw = data[3][:-1]
    trans = data[4][:-1].split(',')
    run = data[5][:-1]
    ret = data[6].split(',')

    remote_folder = '/home/' + user + '/livestock/ssh'

    # Start SSH session
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(ip, port=port, username=user, password=pw)
    print('Opening SSH Connection')

    # Copy files to remove server
    sftp = ssh.open_sftp()
    check_for_remote_folder(sftp, '/home/' + user + '/livestock', 'ssh')

    for f in trans:
        sftp.put(local_folder + '/' + f, remote_folder + '/' + f)
    sftp.put(local_folder + '/in_data.txt', remote_folder + '/in_data.txt')

    channel = ssh.invoke_shell()

    channel_data = ''

    com_send = False
    folder_send = False
    outfile = False

    while True:
        # Print shell
        if channel.recv_ready():
            channel_bytes = channel.recv(9999)
            channel_data += channel_bytes.decode("utf-8")
            print(channel_data)

        else:
            pass

        # Execute commands
        if not folder_send:
            sftp.chdir(remote_folder)
            channel.send('cd ' + remote_folder + '\n')
            print('Folder Send\n')
            folder_send = True

        elif folder_send and not com_send:
            channel.send('source activate livestock_env' + '\n')
            channel.send('python ' + run + '\n')
            print('Command Send\n')
            com_send = True

        else:
            pass

        # Look for outfile
        try:
            outfile = sftp.file(remote_folder + '/out.txt')
        except Exception:
            pass

        if outfile:
            print('Found out file\n')
            sftp.get(remote_folder + '/out.txt', local_folder + '\\out.txt')
            sftp.remove('out.txt')

        # If found start transferring files and clean up
        if os.path.isfile(local_folder + '\\out.txt'):

            # Copy result files to local and delete remotely
            print('Copying and deleting result files:')

            # Get return files
            print('Transferring files:')
            for f in ret:
                print(f)
                sftp.get(remote_folder + '/' + f, local_folder + '/' + f)
                sftp.remove(f)
            print('')

            # Delete input files
            print('Deleting remote files:')
            for f in sftp.listdir():
                print(f)
                sftp.remove(f)

            print('')
            break

        else:
            pass

    # Close connection
    print('Closing SSH Connection!')
    sftp.close()
    ssh.close()
