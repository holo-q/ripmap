/**
 * Sample JavaScript file for testing tag extraction.
 */

// Constants
const API_VERSION = '2.0';
const MAX_TIMEOUT = 30000;

class UserService {
    constructor(apiClient) {
        this.apiClient = apiClient;
    }
    
    async fetchUser(userId) {
        return await this.apiClient.get(`/users/${userId}`);
    }
    
    updateProfile(userId, data) {
        return this.apiClient.post(`/users/${userId}`, data);
    }
}

function validateEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

const parseDate = (dateString) => {
    return new Date(dateString);
};

export { UserService, validateEmail, parseDate };
