import axios from 'axios';

const useAxios = () => {
    const apiBaseURL = `http://10.147.18.156:8000/api`;

    const axiosInstance = axios.create({
        baseURL: apiBaseURL,
    });

    return axiosInstance;
};

export default useAxios;
