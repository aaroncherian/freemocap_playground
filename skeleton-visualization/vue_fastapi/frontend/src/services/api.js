import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:8000',
  withCredentials: false,
  headers: {
    Accept: 'application/json',
    'Content-Type': 'application/json'
  }
});

export default {
  getAvailableJointNames() {
    return apiClient.get('/available_joint_names');
  },
  getData() {
    return apiClient.get('/data');
  }
};