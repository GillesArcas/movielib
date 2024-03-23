/*
Firefox localStorage:
C:\Users\<user>\AppData\Roaming\Mozilla\Firefox\Profiles\2t7nsv81.default-release-1603874366686\storage\default
*/

function historyDict() {
    let movies_to_dates = {};
    for (const movie of Object.keys(localStorage)) {
        const value = localStorage.getItem(movie);
        if (value.search(/\d\d\d\d-\d\d-\d\d/) >= 0)
            movies_to_dates[movie] = value;
    }
    return movies_to_dates;
}

function set_history_status(value) {
    // value is 'saved' or 'modified'
    document.cookie = `history_status=${value};samesite=lax;max-age=31536000;path=/`;
}

function get_history_status() {
    // return 'saved' or 'modified'
    for (const pair of document.cookie.split("; ")) {
        if (pair.startsWith('history_status'))
            return pair.split("=")[1];
    }
    return 'saved';  // first utilisation or cookies have been reset
}

function date_iso_to_loc(dateiso) {
    const options = {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
    };
    return new Date(dateiso).toLocaleDateString("fr-FR", options);
}

function history_ul(dates) {
    s = "<ul style='margin-top:2px;'>\n";
    if (dates !== null && dates != 'null') {
        for (const dateiso of dates.split(',')) {
            s += '<li>' + date_iso_to_loc(dateiso) + '</li>\n';
        }
    }
    s += "<li><a href='javascript:add_date();' title=\"{{T('Add as viewed today')}}\">...</a></li>\n";
    s += "</ul>\n";
    return s;
}

function history_sorted_by_movie() {
    let s = '';
    let movies_to_dates = historyDict();
    for (const movie of Object.keys(movies_to_dates).sort()) {
        let datesloc = movies_to_dates[movie].split(',').map(dateiso => date_iso_to_loc(dateiso));
        s += '<tr>' +
             '<td class="dates">' + movie + '</td>' +
             '<td class="dates">' + datesloc.join(',') + '</td>' +
             '</tr>';
    }
    return s;
}

function history_sorted_by_date() {
    let dates_to_movies = {};
    let movies_to_dates = historyDict();
    for (const movie of Object.keys(movies_to_dates).sort()) {
        let dates = movies_to_dates[movie].split(',');
        for (const date of dates) {
            if (!(date in dates_to_movies))
                dates_to_movies[date] = [];
            dates_to_movies[date].push(movie);
        }
    }

    let s = '';
    for (const dateiso of Object.keys(dates_to_movies).sort()) {
        s += '<tr>' +
             '<td>' + date_iso_to_loc(dateiso) + '</td>' +
             '<td class="dates">' + dates_to_movies[dateiso][0] + '</td>' +
             '</tr>';
        for (const movie of dates_to_movies[dateiso].slice(1))
            s += '<tr>' +
                 '<td>' + '</td>' +
                 '<td class="dates">' + movie + '</td>' +
                 '</tr>';
    }
    return s;
}

function formatted_history_for_saving() {
    let dates_to_movies = {};
    let movies_to_dates = historyDict();
    for (const movie of Object.keys(movies_to_dates).sort()) {
        let dates = movies_to_dates[movie].split(',');
        for (const date of dates) {
            if (!(date in dates_to_movies))
                dates_to_movies[date] = [];
            dates_to_movies[date].push(movie);
        }
    }
    let list = [];
    for (const dateiso of Object.keys(dates_to_movies).sort())
        for (const movie of dates_to_movies[dateiso])
            list.push(movie, dateiso);
    return list.join('\n');
}

